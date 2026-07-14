// Diagnostic: where does docs/objects/10003.jpg rank when querying with
// docs/test/test.jpg (photo of embossed mark), across models, preprocessing
// profiles, and embedding-pooling variants?
import { readdirSync } from "node:fs";
import { join } from "node:path";
import * as t from "@huggingface/transformers";
import { loadEmbedder, applyProfile, l2Normalize, quantize, scoreAll, topK } from "../docs/js/embedding-core.mjs";

const OBJECTS = "../docs/objects";
const QUERY = "../docs/test/test.jpg";
const TARGET = "10003.jpg";
const EXTS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"]);

// --- experimental profiles (applied after the standard/lineart profiles) ---

function toGrayValues(image) {
  const { width, height, data, channels } = image;
  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0; i < width * height; i += 1) {
    const s = i * channels;
    gray[i] = channels >= 3 ? 0.299 * data[s] + 0.587 * data[s + 1] + 0.114 * data[s + 2] : data[s];
  }
  return gray;
}

function fromGray(image, gray, width, height) {
  const out = new Uint8ClampedArray(width * height * 3);
  for (let i = 0; i < width * height; i += 1) {
    out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = gray[i];
  }
  return new (image.constructor)(out, width, height, 3);
}

/** Otsu threshold → black mark on white. Assumes mark is the minority class. */
function binarize(image) {
  const gray = toGrayValues(image);
  const hist = new Uint32Array(256);
  gray.forEach((v) => hist[Math.round(v)]++);
  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i += 1) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, threshold = 127;
  for (let i = 0; i < 256; i += 1) {
    wB += hist[i];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += i * hist[i];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; threshold = i; }
  }
  let dark = 0;
  gray.forEach((v) => { if (v <= threshold) dark++; });
  const markIsDark = dark <= total - dark;
  const out = new Uint8ClampedArray(total);
  for (let i = 0; i < total; i += 1) {
    const isMark = markIsDark ? gray[i] <= threshold : gray[i] > threshold;
    out[i] = isMark ? 0 : 255;
  }
  return fromGray(image, out, image.width, image.height);
}

/** Sobel edge magnitude, normalized, dark edges on white. */
function edges(image) {
  const gray = toGrayValues(image);
  const { width, height } = image;
  const mag = new Float32Array(width * height);
  let max = 1;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const gx = -gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1]
                 + gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1];
      const gy = -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
                 + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      mag[i] = Math.hypot(gx, gy);
      if (mag[i] > max) max = mag[i];
    }
  }
  const out = new Uint8ClampedArray(width * height);
  for (let i = 0; i < mag.length; i += 1) out[i] = 255 - (mag[i] / max) * 255;
  return fromGray(image, out, width, height);
}

const PROFILES = {
  standard: (img) => applyProfile(img, "standard"),
  lineart: (img) => applyProfile(img, "lineart"),
  binary: (img) => binarize(applyProfile(img, "lineart")),
  edge: (img) => edges(applyProfile(img, "lineart")),
};

// --- embedding variants -----------------------------------------------------

async function embedWith(embedder, image, pooling) {
  const inputs = await embedder.processor(image);
  const outputs = await embedder.model(inputs);
  let vector;
  if (embedder.spec.embedding_source === "image_embeds") {
    vector = Float32Array.from(outputs.image_embeds.data);
  } else {
    const { data, dims } = outputs.last_hidden_state;
    const [, tokens, dim] = dims;
    if (pooling === "mean") {
      vector = new Float32Array(dim);
      for (let tok = 1; tok < tokens; tok += 1) {
        for (let d = 0; d < dim; d += 1) vector[d] += data[tok * dim + d];
      }
      for (let d = 0; d < dim; d += 1) vector[d] /= tokens - 1;
    } else {
      vector = Float32Array.from(data.slice(0, dim));
    }
  }
  return l2Normalize(vector);
}

// --- run ---------------------------------------------------------------------

const filenames = readdirSync(OBJECTS).filter((n) => EXTS.has(n.slice(n.lastIndexOf(".")).toLowerCase())).sort();
const targetRow = filenames.indexOf(TARGET);
const rawImages = [];
for (const name of filenames) rawImages.push(await t.RawImage.read(join(OBJECTS, name)));
const queryRaw = await t.RawImage.read(QUERY);

const CONFIGS = [
  ["dinov2", "cls", ["standard", "lineart", "binary", "edge"]],
  ["dinov2", "mean", ["lineart", "binary", "edge"]],
  ["clip", "-", ["standard", "lineart", "binary", "edge"]],
  ["mobilenet", "-", ["lineart", "binary", "edge"]],
];

for (const [modelKey, pooling, profiles] of CONFIGS) {
  const embedder = await loadEmbedder(t, modelKey);
  const dim = embedder.spec.dim;
  for (const profileName of profiles) {
    const profile = PROFILES[profileName];
    const blob = new Int8Array(filenames.length * dim);
    for (let row = 0; row < filenames.length; row += 1) {
      blob.set(quantize(await embedWith(embedder, profile(rawImages[row]), pooling)), row * dim);
    }
    const query = await embedWith(embedder, profile(queryRaw), pooling);
    const scores = scoreAll(query, blob, filenames.length, dim);
    const order = topK(scores, filenames.length);
    const rank = order.indexOf(targetRow) + 1;
    const top3 = order.slice(0, 3).map((r) => `${filenames[r]}:${scores[r].toFixed(2)}`).join(" ");
    console.log(
      `${modelKey}${pooling === "mean" ? "-mean" : ""}`.padEnd(12) +
      profileName.padEnd(9) +
      `target_rank=${String(rank).padEnd(4)} target_sim=${scores[targetRow].toFixed(3)}  top3: ${top3}`,
    );
  }
  await embedder.model.dispose();
}
