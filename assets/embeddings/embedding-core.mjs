/**
 * embedding-core.mjs — shared embedding logic for build (Node) and search (browser).
 *
 * This module is environment-neutral: it never imports @huggingface/transformers
 * itself. Callers pass the loaded Transformers.js namespace into loadEmbedder(),
 * so Node (node_modules) and the browser (CDN ESM) run the exact same code paths
 * for preprocessing, embedding extraction, quantization, and scoring. That shared
 * path is what guarantees collection vectors and query vectors live in the same
 * embedding space.
 */

/**
 * Closed set of supported models. `key` is what users put in config.yml.
 *
 * Note on `mobilenet`: plain MobileNetV2/V4 ONNX exports on the Hugging Face hub
 * only ship classification logits (no feature output), so the lightweight slot is
 * filled by Apple MobileCLIP-S0, whose image tower is a MobileNet-family network.
 */
export const MODEL_REGISTRY = {
  dinov2: {
    hf_id: "Xenova/dinov2-small",
    dim: 384,
    dtype: "q8",
    embedding_source: "cls_token",
    label: "DINOv2-small",
    approx_download_mb: 24,
  },
  clip: {
    hf_id: "Xenova/clip-vit-base-patch32",
    dim: 512,
    dtype: "q8",
    embedding_source: "image_embeds",
    label: "CLIP ViT-B/32",
    approx_download_mb: 85,
    supports_text: true,
    text_approx_download_mb: 65,
  },
  mobilenet: {
    hf_id: "Xenova/mobileclip_s0",
    dim: 512,
    dtype: "q8",
    embedding_source: "image_embeds",
    label: "MobileCLIP-S0",
    approx_download_mb: 12,
  },
};

export const PREPROCESSING_PROFILES = ["standard", "lineart", "binary"];

/** Fixed int8 quantization scale for L2-normalized vectors (values in [-1, 1]). */
export const QUANT_SCALE = 127;

export function getModelSpec(key) {
  const spec = MODEL_REGISTRY[key];
  if (!spec) {
    const known = Object.keys(MODEL_REGISTRY).join(", ");
    throw new Error(`Unknown model key "${key}". Supported: ${known}`);
  }
  return { key, ...spec };
}

/**
 * Load processor + model for a registry key.
 * @param {object} transformers - the @huggingface/transformers namespace.
 * @param {string} key - registry key (dinov2 | clip | mobilenet).
 * @param {object} [options] - passed through to from_pretrained (e.g.
 *   progress_callback, device).
 * @returns {Promise<{processor, model, spec}>} an "embedder".
 */
export async function loadEmbedder(transformers, key, options = {}) {
  const spec = getModelSpec(key);
  const processor = await transformers.AutoProcessor.from_pretrained(spec.hf_id, options);
  const ModelClass =
    spec.embedding_source === "image_embeds"
      ? transformers.CLIPVisionModelWithProjection
      : transformers.AutoModel;
  const model = await ModelClass.from_pretrained(spec.hf_id, {
    dtype: spec.dtype,
    ...options,
  });
  return { processor, model, spec };
}

/**
 * Load tokenizer + text tower for a registry key with a text encoder.
 * CLIP-family models project text into the same embedding space as images,
 * so a text query can be scored against the existing image vectors.
 * @param {object} transformers - the @huggingface/transformers namespace.
 * @param {string} key - registry key with `supports_text` (currently: clip).
 * @param {object} [options] - passed through to from_pretrained (e.g.
 *   progress_callback, device).
 * @returns {Promise<{tokenizer, textModel, spec}>} a "text embedder".
 */
export async function loadTextEmbedder(transformers, key, options = {}) {
  const spec = getModelSpec(key);
  if (!spec.supports_text) {
    throw new Error(`model "${key}" has no text tower; text search requires a CLIP-family model`);
  }
  const tokenizer = await transformers.AutoTokenizer.from_pretrained(spec.hf_id, options);
  const textModel = await transformers.CLIPTextModelWithProjection.from_pretrained(spec.hf_id, {
    dtype: spec.dtype,
    ...options,
  });
  return { tokenizer, textModel, spec };
}

/**
 * Compute an L2-normalized Float32Array embedding for a free-text query,
 * in the same space as embedImage() vectors for CLIP-family models.
 * @param {{tokenizer, textModel, spec}} textEmbedder - from loadTextEmbedder().
 */
export async function embedText(textEmbedder, query) {
  const { tokenizer, textModel, spec } = textEmbedder;
  const text = (query ?? "").toString().trim();
  if (!text) {
    throw new Error("text query is empty");
  }
  const inputs = tokenizer([text], { padding: true, truncation: true });
  const outputs = await textModel(inputs);
  const vector = outputs.text_embeds.data;
  if (vector.length !== spec.dim) {
    throw new Error(`expected ${spec.dim}-dim text embedding, got ${vector.length}`);
  }
  return l2Normalize(Float32Array.from(vector));
}

/**
 * Apply a preprocessing profile to a RawImage, returning a new 3-channel
 * RawImage. Runs before the model's own processor on both build and query
 * sides so the two stay in lockstep.
 */
export function applyProfile(rawImage, profile) {
  if (!PREPROCESSING_PROFILES.includes(profile)) {
    throw new Error(`Unknown preprocessing profile "${profile}". Supported: ${PREPROCESSING_PROFILES.join(", ")}`);
  }
  let image = flattenToWhite(rawImage);
  if (profile === "lineart" || profile === "binary") {
    image = contrastStretchGray(image);
  }
  image = padToSquare(image);
  if (profile === "binary") {
    image = binarizeOtsu(image);
  }
  return image;
}

/** Composite transparency over white and return an RGB image. */
function flattenToWhite(rawImage) {
  const { width, height, channels, data } = rawImage;
  const out = new Uint8ClampedArray(width * height * 3);
  const pixels = width * height;
  for (let i = 0; i < pixels; i += 1) {
    const src = i * channels;
    const dst = i * 3;
    if (channels === 4) {
      const alpha = data[src + 3] / 255;
      out[dst] = data[src] * alpha + 255 * (1 - alpha);
      out[dst + 1] = data[src + 1] * alpha + 255 * (1 - alpha);
      out[dst + 2] = data[src + 2] * alpha + 255 * (1 - alpha);
    } else if (channels === 1) {
      out[dst] = out[dst + 1] = out[dst + 2] = data[src];
    } else if (channels === 2) {
      const alpha = data[src + 1] / 255;
      const value = data[src] * alpha + 255 * (1 - alpha);
      out[dst] = out[dst + 1] = out[dst + 2] = value;
    } else {
      out[dst] = data[src];
      out[dst + 1] = data[src + 1];
      out[dst + 2] = data[src + 2];
    }
  }
  return newImageLike(rawImage, out, width, height);
}

/**
 * Grayscale (Rec. 601 luma) then linearly stretch the 2nd–98th percentile
 * range to full contrast. Normalizes faded scans and photographed marks.
 */
function contrastStretchGray(rgbImage) {
  const { width, height, data } = rgbImage;
  const pixels = width * height;
  const gray = new Uint8ClampedArray(pixels);
  const histogram = new Uint32Array(256);
  for (let i = 0; i < pixels; i += 1) {
    const src = i * 3;
    const value = Math.round(0.299 * data[src] + 0.587 * data[src + 1] + 0.114 * data[src + 2]);
    gray[i] = value;
    histogram[value] += 1;
  }

  const lowTarget = pixels * 0.02;
  const highTarget = pixels * 0.98;
  let low = 0;
  let high = 255;
  let cumulative = 0;
  for (let value = 0; value < 256; value += 1) {
    cumulative += histogram[value];
    if (cumulative >= lowTarget) {
      low = value;
      break;
    }
  }
  cumulative = 0;
  for (let value = 0; value < 256; value += 1) {
    cumulative += histogram[value];
    if (cumulative >= highTarget) {
      high = value;
      break;
    }
  }

  const range = Math.max(1, high - low);
  const out = new Uint8ClampedArray(pixels * 3);
  for (let i = 0; i < pixels; i += 1) {
    const stretched = ((gray[i] - low) * 255) / range;
    const dst = i * 3;
    out[dst] = out[dst + 1] = out[dst + 2] = stretched;
  }
  return newImageLike(rgbImage, out, width, height);
}

/**
 * Otsu-threshold to a pure black-mark-on-white image (inverting if the mark
 * is lighter than its background). Collapses different renderings of the
 * same mark — inked print, blind emboss, photo, scan — into one
 * representation, which measurably improves cross-domain retrieval.
 */
function binarizeOtsu(rgbImage) {
  const { width, height, data } = rgbImage;
  const pixels = width * height;
  const gray = new Uint8ClampedArray(pixels);
  const histogram = new Uint32Array(256);
  for (let i = 0; i < pixels; i += 1) {
    const src = i * 3;
    const value = Math.round(0.299 * data[src] + 0.587 * data[src + 1] + 0.114 * data[src + 2]);
    gray[i] = value;
    histogram[value] += 1;
  }

  let totalSum = 0;
  for (let value = 0; value < 256; value += 1) {
    totalSum += value * histogram[value];
  }
  let sumBelow = 0;
  let weightBelow = 0;
  let bestVariance = 0;
  let threshold = 127;
  for (let value = 0; value < 256; value += 1) {
    weightBelow += histogram[value];
    if (weightBelow === 0) continue;
    const weightAbove = pixels - weightBelow;
    if (weightAbove === 0) break;
    sumBelow += value * histogram[value];
    const meanBelow = sumBelow / weightBelow;
    const meanAbove = (totalSum - sumBelow) / weightAbove;
    const betweenVariance = weightBelow * weightAbove * (meanBelow - meanAbove) ** 2;
    if (betweenVariance > bestVariance) {
      bestVariance = betweenVariance;
      threshold = value;
    }
  }

  // The mark is assumed to be the minority class of pixels.
  let darkCount = 0;
  for (let i = 0; i < pixels; i += 1) {
    if (gray[i] <= threshold) darkCount += 1;
  }
  const markIsDark = darkCount <= pixels - darkCount;

  const out = new Uint8ClampedArray(pixels * 3);
  for (let i = 0; i < pixels; i += 1) {
    const isMark = markIsDark ? gray[i] <= threshold : gray[i] > threshold;
    const value = isMark ? 0 : 255;
    const dst = i * 3;
    out[dst] = out[dst + 1] = out[dst + 2] = value;
  }
  return newImageLike(rgbImage, out, width, height);
}

/** Pad to a centered square with white fill (no aspect-ratio distortion). */
function padToSquare(rgbImage) {
  const { width, height, data } = rgbImage;
  if (width === height) {
    return rgbImage;
  }
  const size = Math.max(width, height);
  const out = new Uint8ClampedArray(size * size * 3).fill(255);
  const offsetX = Math.floor((size - width) / 2);
  const offsetY = Math.floor((size - height) / 2);
  for (let y = 0; y < height; y += 1) {
    const srcRow = y * width * 3;
    const dstRow = ((y + offsetY) * size + offsetX) * 3;
    out.set(data.subarray(srcRow, srcRow + width * 3), dstRow);
  }
  return newImageLike(rgbImage, out, size, size);
}

/** Construct a new RawImage without importing the class (environment-neutral). */
function newImageLike(rawImage, data, width, height) {
  const ImageClass = rawImage.constructor;
  return new ImageClass(data, width, height, 3);
}

/**
 * Compute an L2-normalized Float32Array embedding for a RawImage.
 * @param {{processor, model, spec}} embedder - from loadEmbedder().
 */
export async function embedImage(embedder, rawImage, profile) {
  const { processor, model, spec } = embedder;
  const image = applyProfile(rawImage, profile);
  const inputs = await processor(image);
  const outputs = await model(inputs);

  let vector;
  if (spec.embedding_source === "image_embeds") {
    vector = outputs.image_embeds.data;
  } else {
    // cls_token: last_hidden_state is [1, tokens, dim]; the CLS token is
    // token 0, so its values are the first `dim` entries.
    vector = outputs.last_hidden_state.data.slice(0, spec.dim);
  }

  if (vector.length !== spec.dim) {
    throw new Error(`expected ${spec.dim}-dim embedding, got ${vector.length}`);
  }
  return l2Normalize(Float32Array.from(vector));
}

export function l2Normalize(vector) {
  let normSq = 0;
  for (let i = 0; i < vector.length; i += 1) {
    normSq += vector[i] * vector[i];
  }
  const norm = Math.sqrt(normSq);
  if (!(norm > 0)) {
    throw new Error("embedding norm is zero");
  }
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    out[i] = vector[i] / norm;
  }
  return out;
}

/** Quantize an L2-normalized vector to int8 at QUANT_SCALE. */
export function quantize(vector) {
  const out = new Int8Array(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    out[i] = Math.max(-QUANT_SCALE, Math.min(QUANT_SCALE, Math.round(vector[i] * QUANT_SCALE)));
  }
  return out;
}

/**
 * Cosine-similarity scores of one query against every row of a quantized
 * collection blob. Both sides are int8 at QUANT_SCALE, so the integer dot
 * product rescales by 1/QUANT_SCALE².
 * @param {Float32Array} queryVector - L2-normalized query embedding.
 * @param {Int8Array} blob - count × dim int8 values, row-major.
 * @returns {Float32Array} scores, one per row, in blob order.
 */
export function scoreAll(queryVector, blob, count, dim) {
  if (queryVector.length !== dim) {
    throw new Error(`query dim ${queryVector.length} does not match collection dim ${dim}`);
  }
  if (blob.length !== count * dim) {
    throw new Error(`blob length ${blob.length} does not match count ${count} × dim ${dim}`);
  }
  const query = quantize(queryVector);
  const scores = new Float32Array(count);
  const rescale = 1 / (QUANT_SCALE * QUANT_SCALE);
  for (let row = 0; row < count; row += 1) {
    let dot = 0;
    const base = row * dim;
    for (let i = 0; i < dim; i += 1) {
      dot += query[i] * blob[base + i];
    }
    scores[row] = dot * rescale;
  }
  return scores;
}

/** Indices of the top-k scores, descending (stable for equal scores). */
export function topK(scores, k) {
  const order = Array.from(scores.keys());
  order.sort((a, b) => scores[b] - scores[a] || a - b);
  return order.slice(0, Math.min(k, order.length));
}

/**
 * Map a raw cosine score to a 0–100 display percentage using the collection's
 * build-time calibration (floor = low percentile of pairwise similarities).
 */
export function calibratedPercent(score, calibration) {
  const { floor, ceiling } = calibration;
  const range = Math.max(1e-6, ceiling - floor);
  const fraction = (score - floor) / range;
  return Math.max(0, Math.min(100, fraction * 100));
}
