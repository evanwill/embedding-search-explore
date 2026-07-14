#!/usr/bin/env node
/**
 * Retrieval evaluation harness: measures how well each model + preprocessing
 * profile retrieves a collection image from a perturbed copy of itself
 * (simulating a user's photo or scan of a known mark).
 *
 * For every evaluated config the full collection is embedded as the search
 * index; sampled images are perturbed (crop, rotation, photo-like degradation)
 * and used as queries. Reported metrics are top-1 / top-5 self-retrieval:
 * the fraction of queries whose source image ranks first / in the top five.
 *
 * Usage:
 *   node eval_retrieval.mjs [--models dinov2,clip,mobilenet]
 *                           [--profiles standard,lineart]
 *                           [--sample 65]
 *
 * Note: filename variants (e.g. 10139.png and 10139-00.png) are treated as
 * distinct, so scores are conservative — a "miss" may be a near-duplicate.
 */

import { readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as transformers from "@huggingface/transformers";
import { parse as parseYaml } from "yaml";
import { readFileSync } from "node:fs";

import {
  MODEL_REGISTRY,
  PREPROCESSING_PROFILES,
  loadEmbedder,
  embedImage,
  quantize,
  scoreAll,
  topK,
} from "../docs/js/embedding-core.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"]);

function parseArgs() {
  const args = { models: Object.keys(MODEL_REGISTRY), profiles: [...PREPROCESSING_PROFILES], sample: 65 };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--models") args.models = argv[++i].split(",");
    else if (argv[i] === "--profiles") args.profiles = argv[++i].split(",");
    else if (argv[i] === "--sample") args.sample = Number(argv[++i]);
  }
  return args;
}

// --- perturbations (RawImage → RawImage, all on RGB data) -----------------

function toRgb(image) {
  return image.channels === 3 ? image : image.rgb();
}

function newImageLike(image, data, width, height) {
  const ImageClass = image.constructor;
  return new ImageClass(data, width, height, 3);
}

/** Crop `fraction` off every border. */
function crop(image, fraction = 0.1) {
  const src = toRgb(image);
  const marginX = Math.floor(src.width * fraction);
  const marginY = Math.floor(src.height * fraction);
  const width = src.width - 2 * marginX;
  const height = src.height - 2 * marginY;
  const out = new Uint8ClampedArray(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const srcStart = ((y + marginY) * src.width + marginX) * 3;
    out.set(src.data.subarray(srcStart, srcStart + width * 3), y * width * 3);
  }
  return newImageLike(src, out, width, height);
}

/** Rotate by `degrees` around the center, white fill, nearest-neighbor. */
function rotate(image, degrees = 5) {
  const src = toRgb(image);
  const { width, height, data } = src;
  const radians = (degrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const cx = width / 2;
  const cy = height / 2;
  const out = new Uint8ClampedArray(width * height * 3).fill(255);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const sx = Math.round(cx + dx * cos + dy * sin);
      const sy = Math.round(cy - dx * sin + dy * cos);
      if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        const srcIdx = (sy * width + sx) * 3;
        const dstIdx = (y * width + x) * 3;
        out[dstIdx] = data[srcIdx];
        out[dstIdx + 1] = data[srcIdx + 1];
        out[dstIdx + 2] = data[srcIdx + 2];
      }
    }
  }
  return newImageLike(src, out, width, height);
}

/** Photo-like degradation: dim, reduce contrast, add noise (deterministic). */
function photo(image) {
  const src = toRgb(image);
  const out = new Uint8ClampedArray(src.data.length);
  let seed = 12345;
  const random = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < src.data.length; i += 1) {
    const dimmed = src.data[i] * 0.9;
    const flattened = (dimmed - 128) * 0.85 + 140;
    out[i] = flattened + (random() - 0.5) * 24;
  }
  return newImageLike(src, out, src.width, src.height);
}

const PERTURBATIONS = { crop10: crop, rotate5: rotate, photo };

// --- evaluation ------------------------------------------------------------

async function evaluateConfig(modelKey, profile, filenames, queryFilenames, objectsDir) {
  const embedder = await loadEmbedder(transformers, modelKey);
  const dim = embedder.spec.dim;

  const blob = new Int8Array(filenames.length * dim);
  const rawImages = new Map();
  for (const [row, filename] of filenames.entries()) {
    const image = await transformers.RawImage.read(join(objectsDir, filename));
    if (queryFilenames.has(filename)) {
      rawImages.set(filename, image);
    }
    blob.set(quantize(await embedImage(embedder, image, profile)), row * dim);
  }

  const results = {};
  for (const [name, perturb] of Object.entries(PERTURBATIONS)) {
    let top1 = 0;
    let top5 = 0;
    for (const filename of queryFilenames) {
      const query = await embedImage(embedder, perturb(rawImages.get(filename)), profile);
      const scores = scoreAll(query, blob, filenames.length, dim);
      const ranked = topK(scores, 5).map((row) => filenames[row]);
      if (ranked[0] === filename) top1 += 1;
      if (ranked.includes(filename)) top5 += 1;
    }
    results[name] = { top1: top1 / queryFilenames.size, top5: top5 / queryFilenames.size };
  }

  await embedder.model.dispose();
  return results;
}

async function main() {
  const args = parseArgs();
  const config = parseYaml(readFileSync(join(PROJECT_ROOT, "config.yml"), "utf8"));
  const objectsDir = resolve(PROJECT_ROOT, config.objects_dir ?? "docs/objects");

  const filenames = readdirSync(objectsDir)
    .filter((name) => SUPPORTED_EXTENSIONS.has(name.slice(name.lastIndexOf(".")).toLowerCase()))
    .sort();

  const step = Math.max(1, Math.floor(filenames.length / args.sample));
  const queryFilenames = new Set(filenames.filter((_, i) => i % step === 0).slice(0, args.sample));

  console.log(`collection=${filenames.length} images, queries=${queryFilenames.size}, perturbations=${Object.keys(PERTURBATIONS).join("/")}`);
  console.log("");

  const rows = [];
  for (const modelKey of args.models) {
    for (const profile of args.profiles) {
      const started = Date.now();
      const results = await evaluateConfig(modelKey, profile, filenames, queryFilenames, objectsDir);
      const elapsed = ((Date.now() - started) / 1000).toFixed(0);
      for (const [perturbation, { top1, top5 }] of Object.entries(results)) {
        rows.push({ model: modelKey, profile, perturbation, top1, top5 });
      }
      const summary = Object.entries(results)
        .map(([name, { top1, top5 }]) => `${name} top1=${(top1 * 100).toFixed(0)}% top5=${(top5 * 100).toFixed(0)}%`)
        .join("  ");
      console.log(`${modelKey.padEnd(10)} ${profile.padEnd(9)} ${summary}  (${elapsed}s)`);
    }
  }

  console.log("");
  console.log("| model | profile | perturbation | top-1 | top-5 |");
  console.log("|---|---|---|---|---|");
  for (const row of rows) {
    console.log(
      `| ${row.model} | ${row.profile} | ${row.perturbation} | ${(row.top1 * 100).toFixed(0)}% | ${(row.top5 * 100).toFixed(0)}% |`,
    );
  }
}

main().catch((error) => {
  console.error(`Eval failed: ${error.message}`);
  process.exit(1);
});
