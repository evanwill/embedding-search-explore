#!/usr/bin/env node
/**
 * Build browser-ready image embedding artifacts for the reverse lookup app.
 *
 * Reads config.yml at the repo root, embeds every collection image with the
 * configured model + preprocessing profile (via the same shared module the
 * browser uses), and writes to the configured output dir:
 *   - manifest.json    build configuration export read by the search page
 *   - embeddings.bin   count × dim int8 L2-normalized vectors, row-major
 *   - index.json       trimmed metadata records for result cards
 *   - build-info.json  counts, skips, unmatched metadata rows
 *   - preprocess.log   plain-text build summary
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
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
} from "../docs/js/embedding-core.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMAS_DIR = join(PROJECT_ROOT, "scripts", "schemas");
const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"]);
const CARD_FIELDS = ["title", "active_years", "printers", "publishers", "website"];
const CALIBRATION_FLOOR_PERCENTILE = 0.05;

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function loadConfig() {
  const configPath = join(PROJECT_ROOT, "config.yml");
  if (!existsSync(configPath)) {
    fail(`missing ${configPath}`);
  }
  const config = parseYaml(readFileSync(configPath, "utf8"));

  const modelKey = config.model ?? "dinov2";
  if (!MODEL_REGISTRY[modelKey]) {
    fail(`config.yml model "${modelKey}" is not one of: ${Object.keys(MODEL_REGISTRY).join(", ")}`);
  }
  const preprocessing = config.preprocessing ?? "standard";
  if (!PREPROCESSING_PROFILES.includes(preprocessing)) {
    fail(`config.yml preprocessing "${preprocessing}" is not one of: ${PREPROCESSING_PROFILES.join(", ")}`);
  }

  return {
    modelKey,
    preprocessing,
    objectsDir: resolve(PROJECT_ROOT, config.objects_dir ?? "docs/objects"),
    metadataCsv: resolve(PROJECT_ROOT, config.metadata_csv ?? "docs/metadata/cb-pmcarchive.csv"),
    filenameField: config.filename_field ?? "filename",
    outputDir: resolve(PROJECT_ROOT, config.output_dir ?? "docs/data"),
    topK: Number(config.top_k ?? 12),
  };
}

function loadMetadata(csvPath, filenameField) {
  const rows = parseCsv(readFileSync(csvPath, "utf8"), {
    bom: true,
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });
  const byFilename = new Map();
  for (const row of rows) {
    const filename = (row[filenameField] ?? "").trim();
    if (filename) {
      byFilename.set(filename, row);
    }
  }
  return { rows, byFilename };
}

function listImages(objectsDir) {
  return readdirSync(objectsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && SUPPORTED_EXTENSIONS.has(extensionOf(entry.name)))
    .map((entry) => entry.name)
    .sort();
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

function indexRecord(filename, metadataRow) {
  const record = {
    filename,
    image_path: relativizePath(metadataRow.object_location) || `objects/${filename}`,
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

  const { rows: metadataRows, byFilename } = loadMetadata(config.metadataCsv, config.filenameField);
  const images = listImages(config.objectsDir);

  console.log(`Model: ${spec.label} (${spec.hf_id}), preprocessing: ${config.preprocessing}`);
  console.log("Loading model (first run downloads weights; they are cached for later runs) ...");
  const embedder = await loadEmbedder(transformers, config.modelKey);

  const generatedAt = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  const skipped = [];
  const filenames = [];
  const quantizedRows = [];
  const indexItems = [];

  let done = 0;
  for (const filename of images) {
    done += 1;
    const metadataRow = byFilename.get(filename);
    if (!metadataRow) {
      skipped.push({ filename, reason: "missing metadata row for filename" });
      continue;
    }
    try {
      const image = await transformers.RawImage.read(join(config.objectsDir, filename));
      const vector = await embedImage(embedder, image, config.preprocessing);
      filenames.push(filename);
      quantizedRows.push(quantize(vector));
      indexItems.push(indexRecord(filename, metadataRow));
    } catch (error) {
      skipped.push({ filename, reason: `embedding failed: ${error.message}` });
      continue;
    }
    if (done % 25 === 0 || done === images.length) {
      console.log(`  embedded ${done}/${images.length}`);
    }
  }

  const count = filenames.length;
  const blob = new Int8Array(count * spec.dim);
  quantizedRows.forEach((row, i) => blob.set(row, i * spec.dim));

  const matched = new Set(filenames);
  const unmatchedMetadata = metadataRows
    .map((row) => (row[config.filenameField] ?? "").trim())
    .filter((name) => name && !matched.has(name))
    .sort();

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
      images_found: images.length,
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
    `images_found=${images.length}`,
    `metadata_rows=${metadataRows.length}`,
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
    `Summary: images_found=${images.length}, embeddings_built=${count}, skipped_images=${skipped.length}`,
  );
}

main().catch((error) => {
  console.error(`Build failed: ${error.message}`);
  process.exit(1);
});
