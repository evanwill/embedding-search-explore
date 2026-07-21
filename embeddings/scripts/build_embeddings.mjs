#!/usr/bin/env node
/**
 * Build browser-ready image embedding artifacts for the reverse lookup app.
 *
 * Reads embeddings/config-embeddings.yml for embedding settings plus
 * _config.yml metadata pointer, embeds every collection image with the
 * configured model + preprocessing profile (via the same shared module the
 * browser uses), and writes to the configured output dir:
 *   - manifest.json    build configuration export read by the search page
 *   - embeddings.bin   count × dim int8 L2-normalized vectors, row-major
 *   - index.json       trimmed metadata records for result cards
 *   - build-info.json  counts, skips, unmatched metadata rows
 *   - preprocess.log   plain-text build summary
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as transformers from "@huggingface/transformers";
import { parse as parseCsv } from "csv-parse/sync";
import { parse as parseYaml } from "yaml";
import Ajv2020 from "ajv/dist/2020.js";

import {
  MODEL_REGISTRY,
  PREPROCESSING_PROFILES,
  QUANT_SCALE,
  getModelSpec,
  loadEmbedder,
  embedImage,
  quantize,
} from "../../assets/embeddings/embedding-core.mjs";

const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"]);
const CARD_FIELDS = ["title"];
const CALIBRATION_FLOOR_PERCENTILE = 0.05;
const EMBEDDINGS_CONFIG_FILE = "config-embeddings.yml";
const COLLECTION_CONFIG_FILE = "_config.yml";

const EMBEDDINGS_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO_ROOT = resolve(EMBEDDINGS_ROOT, "..");
const SCHEMAS_DIR = join(EMBEDDINGS_ROOT, "scripts", "schemas");
const EMBEDDINGS_CONFIG_PATH = join(EMBEDDINGS_ROOT, EMBEDDINGS_CONFIG_FILE);
const COLLECTION_CONFIG_PATH = join(REPO_ROOT, COLLECTION_CONFIG_FILE);

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function loadYaml(path, label) {
  if (!existsSync(path)) {
    fail(`missing ${label}: ${path}`);
  }
  return parseYaml(readFileSync(path, "utf8"));
}

function resolveFromRepo(pathValue) {
  return resolve(REPO_ROOT, pathValue);
}

function normalizeText(value) {
  return (value ?? "").toString().trim();
}

function normalizeLower(value) {
  return normalizeText(value).toLowerCase();
}

function loadConfig() {
  const embeddingConfig = loadYaml(EMBEDDINGS_CONFIG_PATH, EMBEDDINGS_CONFIG_FILE);
  const collectionConfig = loadYaml(COLLECTION_CONFIG_PATH, COLLECTION_CONFIG_FILE);

  const metadataKey = (collectionConfig.metadata ?? "").toString().trim();
  if (!metadataKey) {
    fail(`${COLLECTION_CONFIG_FILE} is missing required "metadata" key`);
  }

  const modelKey = embeddingConfig.model ?? "dinov2";
  if (!MODEL_REGISTRY[modelKey]) {
    fail(`${EMBEDDINGS_CONFIG_FILE} model "${modelKey}" is not one of: ${Object.keys(MODEL_REGISTRY).join(", ")}`);
  }

  const preprocessing = embeddingConfig.preprocessing ?? "standard";
  if (!PREPROCESSING_PROFILES.includes(preprocessing)) {
    fail(`${EMBEDDINGS_CONFIG_FILE} preprocessing "${preprocessing}" is not one of: ${PREPROCESSING_PROFILES.join(", ")}`);
  }

  const topK = Number(embeddingConfig.top_k ?? 12);
  if (!Number.isFinite(topK) || topK < 1) {
    fail(`${EMBEDDINGS_CONFIG_FILE} top_k must be a positive number`);
  }

  return {
    modelKey,
    preprocessing,
    objectsDir: resolveFromRepo(embeddingConfig.objects_dir ?? "objects"),
    metadataCsv: resolveFromRepo(join("_data", `${metadataKey}.csv`)),
    filenameField: embeddingConfig.filename_field ?? "filename",
    outputDir: resolveFromRepo(embeddingConfig.output_dir ?? "assets/embeddings/data"),
    topK,
  };
}

function loadMetadata(csvPath, filenameField) {
  const rows = parseCsv(readFileSync(csvPath, "utf8"), {
    bom: true,
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });

  if (!rows.length) {
    fail(`metadata CSV has no rows: ${csvPath}`);
  }

  const requiredColumns = ["objectid", "display_template", "object_location"];
  const header = rows[0];
  const missingColumns = requiredColumns.filter((key) => !Object.prototype.hasOwnProperty.call(header, key));
  if (missingColumns.length) {
    fail(`metadata CSV is missing required column(s): ${missingColumns.join(", ")}`);
  }

  const imageRows = [];
  const skippedRows = [];
  const seenFilenames = new Set();
  let nonImageRows = 0;

  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const rowNumber = index + 2;

    const objectid = normalizeText(row.objectid);
    if (!objectid) {
      fail(`metadata row ${rowNumber} is missing required value "objectid"`);
    }

    const displayTemplate = normalizeLower(row.display_template);
    if (!displayTemplate) {
      fail(`metadata row ${rowNumber} (${objectid}) is missing required value "display_template"`);
    }
    if (displayTemplate !== "image") {
      nonImageRows += 1;
      continue;
    }

    const objectLocation = normalizeText(row.object_location);
    if (!objectLocation) {
      fail(`metadata row ${rowNumber} (${objectid}) is missing required value "object_location"`);
    }
    if (/^[a-z]+:\/\//i.test(objectLocation)) {
      skippedRows.push({ filename: objectid, reason: `external object_location not supported (${objectLocation})` });
      continue;
    }

    const objectRelative = objectLocation.replace(/^\/+/, "");
    const objectPath = resolveFromRepo(objectRelative);
    if (!existsSync(objectPath)) {
      skippedRows.push({ filename: objectid, reason: `missing object file: ${objectRelative}` });
      continue;
    }

    const filename = normalizeText(row[filenameField]) || basename(objectRelative);
    if (!filename) {
      skippedRows.push({ filename: objectid, reason: `could not derive filename from field "${filenameField}" or object_location` });
      continue;
    }
    if (seenFilenames.has(filename)) {
      skippedRows.push({ filename: objectid, reason: `duplicate filename key "${filename}"; ensure ${filenameField} is unique for image rows` });
      continue;
    }

    if (!SUPPORTED_EXTENSIONS.has(extensionOf(filename))) {
      skippedRows.push({ filename: objectid, reason: `unsupported image extension for "${filename}"` });
      continue;
    }

    seenFilenames.add(filename);
    imageRows.push({ row, filename, objectPath });
  }

  return { rows, imageRows, skippedRows, nonImageRows };
}

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

// Site-relative CSV paths like "/objects/x.png" break when the page is
// hosted from a subpath (e.g. GitHub Pages project sites), so make them
// relative to the page. Full URLs pass through untouched.
function relativizePath(value) {
  const path = (value ?? "").trim();
  if (!path || /^[a-z]+:\/\//i.test(path)) {
    return path;
  }
  return path.replace(/^\/+/, "");
}

function encodePathPart(value) {
  return encodeURIComponent(normalizeText(value));
}

function collectionItemUrl(metadataRow) {
  const objectid = normalizeText(metadataRow.objectid);
  const parentid = normalizeText(metadataRow.parentid);
  const baseId = parentid || objectid;
  const itemPath = `/items/${encodePathPart(baseId)}.html`;
  if (parentid) {
    return `${itemPath}#${encodePathPart(objectid)}`;
  }
  return itemPath;
}

function indexRecord(filename, metadataRow) {
  const record = {
    filename,
    image_path: relativizePath(metadataRow.object_location) || `objects/${filename}`,
    item_url: collectionItemUrl(metadataRow),
  };
  const thumb = relativizePath(metadataRow.image_thumb) || relativizePath(metadataRow.image_small);
  if (thumb) {
    record.thumb_path = thumb;
  }
  for (const field of CARD_FIELDS) {
    const value = (metadataRow[field] ?? "").trim();
    if (value) {
      record[field] = value;
    }
  }
  return record;
}

/**
 * Floor for score display calibration: a low percentile of the collection's
 * pairwise cosine similarities, i.e. what "unrelated" looks like in this
 * embedding space. Computed on the quantized vectors the browser will score.
 */
function scoreCalibration(quantizedRows, dim) {
  const count = quantizedRows.length;
  if (count < 2) {
    return { floor: 0, ceiling: 1 };
  }
  const sims = [];
  const rescale = 1 / (QUANT_SCALE * QUANT_SCALE);
  for (let a = 0; a < count; a += 1) {
    for (let b = a + 1; b < count; b += 1) {
      let dot = 0;
      const rowA = quantizedRows[a];
      const rowB = quantizedRows[b];
      for (let i = 0; i < dim; i += 1) {
        dot += rowA[i] * rowB[i];
      }
      sims.push(dot * rescale);
    }
  }
  sims.sort((a, b) => a - b);
  const floor = sims[Math.floor(sims.length * CALIBRATION_FLOOR_PERCENTILE)];
  return { floor: Number(floor.toFixed(6)), ceiling: 1 };
}

function validateArtifact(payload, schemaFile, label) {
  const schema = JSON.parse(readFileSync(join(SCHEMAS_DIR, schemaFile), "utf8"));
  const ajv = new Ajv2020({ allErrors: false, formats: { "date-time": true } });
  const validate = ajv.compile(schema);
  if (!validate(payload)) {
    const first = validate.errors?.[0];
    fail(`schema validation failed for ${label} at ${first?.instancePath || "<root>"}: ${first?.message}`);
  }
}

function writeJson(path, payload) {
  writeFileSync(path, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function main() {
  const config = loadConfig();
  const spec = getModelSpec(config.modelKey);

  if (!existsSync(config.objectsDir)) {
    fail(`missing objects directory: ${config.objectsDir}`);
  }
  if (!existsSync(config.metadataCsv)) {
    fail(`missing metadata CSV: ${config.metadataCsv}`);
  }
  mkdirSync(config.outputDir, { recursive: true });

  const { rows: metadataRows, imageRows, skippedRows, nonImageRows } = loadMetadata(
    config.metadataCsv,
    config.filenameField,
  );

  if (!imageRows.length) {
    fail("no image rows were selected from metadata; check display_template and object_location values");
  }

  console.log(`Model: ${spec.label} (${spec.hf_id}), preprocessing: ${config.preprocessing}`);
  console.log(
    `Metadata rows: ${metadataRows.length}; image rows selected: ${imageRows.length}; non-image rows skipped by filter: ${nonImageRows}`,
  );
  console.log("Loading model (first run downloads weights; they are cached for later runs) ...");
  const embedder = await loadEmbedder(transformers, config.modelKey);

  const generatedAt = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  const skipped = [];
  const filenames = [];
  const quantizedRows = [];
  const indexItems = [];

  for (const entry of skippedRows) {
    skipped.push(entry);
    console.warn(`Warning: skipped ${entry.filename} - ${entry.reason}`);
  }

  let done = 0;
  for (const imageRow of imageRows) {
    done += 1;
    const { filename, objectPath, row: metadataRow } = imageRow;
    try {
      const image = await transformers.RawImage.read(objectPath);
      const vector = await embedImage(embedder, image, config.preprocessing);
      filenames.push(filename);
      quantizedRows.push(quantize(vector));
      indexItems.push(indexRecord(filename, metadataRow));
    } catch (error) {
      skipped.push({ filename, reason: `embedding failed: ${error.message}` });
      console.warn(`Warning: skipped ${filename} - embedding failed: ${error.message}`);
      continue;
    }
    if (done % 25 === 0 || done === imageRows.length) {
      console.log(`  embedded ${done}/${imageRows.length}`);
    }
  }

  const count = filenames.length;
  const blob = new Int8Array(count * spec.dim);
  quantizedRows.forEach((row, i) => blob.set(row, i * spec.dim));

  const unmatchedMetadata = [];

  const manifest = {
    version: 2,
    generated_at: generatedAt,
    library: { name: "@huggingface/transformers", version: transformers.env.version },
    model: {
      key: spec.key,
      hf_id: spec.hf_id,
      dim: spec.dim,
      dtype: spec.dtype,
      embedding_source: spec.embedding_source,
    },
    preprocessing: config.preprocessing,
    embeddings: {
      file: "embeddings.bin",
      count,
      dtype: "int8",
      scale: QUANT_SCALE,
      normalized: "l2",
    },
    filenames,
    score_calibration: scoreCalibration(quantizedRows, spec.dim),
    top_k: config.topK,
  };

  const index = { version: 2, generated_at: generatedAt, items: indexItems };

  const buildInfo = {
    version: 2,
    generated_at: generatedAt,
    config: { model: spec.key, preprocessing: config.preprocessing },
    source: { objects_dir: config.objectsDir, metadata_csv: config.metadataCsv },
    counts: {
      images_found: imageRows.length,
      metadata_rows: metadataRows.length,
      embeddings_built: count,
      skipped_images: skipped.length,
      unmatched_metadata_rows: unmatchedMetadata.length,
    },
    unmatched_metadata_filenames: unmatchedMetadata,
    skipped,
  };

  validateArtifact(manifest, "manifest.schema.json", "manifest.json");
  validateArtifact(index, "index.schema.json", "index.json");
  validateArtifact(buildInfo, "build-info.schema.json", "build-info.json");

  const paths = {
    manifest: join(config.outputDir, "manifest.json"),
    embeddings: join(config.outputDir, "embeddings.bin"),
    index: join(config.outputDir, "index.json"),
    buildInfo: join(config.outputDir, "build-info.json"),
    log: join(config.outputDir, "preprocess.log"),
  };

  writeJson(paths.manifest, manifest);
  writeFileSync(paths.embeddings, Buffer.from(blob.buffer, blob.byteOffset, blob.byteLength));
  writeJson(paths.index, index);
  writeJson(paths.buildInfo, buildInfo);

  const logLines = [
    `generated_at=${generatedAt}`,
    `model=${spec.key} (${spec.hf_id})`,
    `preprocessing=${config.preprocessing}`,
    `images_found=${imageRows.length}`,
    `metadata_rows=${metadataRows.length}`,
    `non_image_rows_filtered=${nonImageRows}`,
    `embeddings_built=${count}`,
    `skipped_images=${skipped.length}`,
    `unmatched_metadata_rows=${unmatchedMetadata.length}`,
    "",
  ];
  if (skipped.length) {
    logLines.push("skipped_images_detail:");
    logLines.push(...skipped.map((entry) => `- ${entry.filename}: ${entry.reason}`));
    logLines.push("");
  }
  if (unmatchedMetadata.length) {
    logLines.push("unmatched_metadata_filenames:");
    logLines.push(...unmatchedMetadata.map((name) => `- ${name}`));
  }
  writeFileSync(paths.log, `${logLines.join("\n")}\n`, "utf8");

  for (const path of Object.values(paths)) {
    console.log(`Wrote ${path}`);
  }
  console.log(
    `Summary: images_found=${imageRows.length}, embeddings_built=${count}, skipped_images=${skipped.length}`,
  );
}

main().catch((error) => {
  console.error(`Build failed: ${error.message}`);
  process.exit(1);
});
