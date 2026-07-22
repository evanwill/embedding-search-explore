#!/usr/bin/env node
/**
 * Smoke test for text-to-image search: embeds free-text queries with the
 * CLIP text tower and ranks them against the built embeddings.bin, printing
 * the top matches per query. Validates the text scoring path end-to-end and
 * gives a quick retrieval-quality read on the collection — run it after
 * `rake build_embeddings` and before trusting the browser text search.
 *
 * Usage:
 *   node text_search_smoke.mjs                       # built-in sample queries
 *   node text_search_smoke.mjs "ore cart" "ladder"   # your own queries
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as transformers from "@huggingface/transformers";

import {
  loadTextEmbedder,
  embedText,
  scoreAll,
  topK,
} from "../../assets/embeddings/embedding-core.mjs";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DATA_DIR = join(REPO_ROOT, "assets", "embeddings", "data");
const TOP_N = 5;

const DEFAULT_QUERIES = [
  "mine tunnel interior",
  "wooden support timbers",
  "men standing in a group",
  "ore cart on rails",
  "ladder",
  "rock wall",
];

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(DATA_DIR, "manifest.json"), "utf8"));
if (!manifest.text_search?.available && !manifest.model?.key) {
  fail("manifest.json not found or unreadable — run `rake build_embeddings` first");
}

const blob = new Int8Array(readFileSync(join(DATA_DIR, "embeddings.bin")).buffer.slice(0));
const index = JSON.parse(readFileSync(join(DATA_DIR, "index.json"), "utf8"));
const titleByFilename = new Map(index.items.map((item) => [item.filename, item.title || item.filename]));

const queries = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_QUERIES;

console.log(`Model: ${manifest.model.hf_id}; collection: ${manifest.embeddings.count} images, "${manifest.preprocessing}" preprocessing`);
console.log("Loading text tower (first run downloads weights; they are cached for later runs) ...");
const textEmbedder = await loadTextEmbedder(transformers, manifest.model.key);

for (const query of queries) {
  const vector = await embedText(textEmbedder, query);
  const scores = scoreAll(vector, blob, manifest.embeddings.count, manifest.model.dim);
  const rows = topK(scores, TOP_N);
  console.log(`\n"${query}"`);
  for (const row of rows) {
    const filename = manifest.filenames[row];
    console.log(`  ${scores[row].toFixed(4)}  ${filename}  ${titleByFilename.get(filename) ?? ""}`);
  }
}
