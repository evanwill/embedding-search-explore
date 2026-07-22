/**
 * Reverse image lookup search page.
 *
 * Configures itself entirely from data/manifest.json (written by the build):
 * which model to load, which preprocessing profile to apply, quantization
 * parameters, and score calibration. Rebuilding with different settings in
 * embeddings/config-embeddings.yml updates this page with no code edits.
 */

import * as transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import {
  getModelSpec,
  loadEmbedder,
  embedImage,
  scoreAll,
  topK,
  calibratedPercent,
} from "./embedding-core.mjs";

const appRoot = document.getElementById("reverse-lookup-app");
const DATA_BASE = normalizeBase(appRoot?.dataset.dataBase || "assets/embeddings/data/");
const SITE_ROOT = normalizeSiteRoot(appRoot?.dataset.siteRoot || "/");

const introCard = document.getElementById("rl-intro");
const introStatus = document.getElementById("rl-intro-status");
const downloadNote = document.getElementById("rl-download-note");
const startButton = document.getElementById("rl-start-button");
const searchUi = document.getElementById("rl-search-ui");
const resultsSection = document.getElementById("rl-results-section");
const imageInput = document.getElementById("rl-image-input");
const searchButton = document.getElementById("rl-search-button");
const statusBox = document.getElementById("rl-status");
const progressWrap = document.getElementById("rl-progress-wrap");
const progressBar = document.getElementById("rl-progress");
const previewWrap = document.getElementById("rl-preview-wrap");
const previewImage = document.getElementById("rl-preview");
const resultsMeta = document.getElementById("rl-results-meta");
const resultsGrid = document.getElementById("rl-results");

let manifest = null;
let blob = null;
let indexByFilename = new Map();
let embedder = null;
let selectedFile = null;
let activeDevice = "wasm";

// Minimum acceptable self-similarity when the backend self-check re-embeds a
// known indexed image and scores it against its own stored vector. A working
// backend lands near 1.0 (int8 quantization costs a little); a broken one
// produces garbage vectors that score far lower. A score threshold is robust
// where a rank-1 check is not: collections with near-duplicate images can
// legitimately rank a sibling above the seed within numerical noise.
const SELF_CHECK_MIN_SCORE = 0.95;

function normalizeBase(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function normalizeSiteRoot(value) {
  if (!value) {
    return "/";
  }
  return value.endsWith("/") ? value : `${value}/`;
}

function isExternalUrl(value) {
  return /^[a-z]+:\/\//i.test(value || "");
}

// index.json image/thumb paths are site-root-relative (leading slash
// stripped at build time). Resolve them against data-site-root so they work
// regardless of where this page's permalink lives.
function resolveAssetUrl(path) {
  const value = (path || "").trim();
  if (!value) {
    return "";
  }
  if (isExternalUrl(value)) {
    return value;
  }
  return `${SITE_ROOT}${value.replace(/^\/+/, "")}`;
}

function resolveItemUrl(item) {
  const path = (item?.item_url || "").trim();
  if (!path) {
    return "";
  }
  if (isExternalUrl(path)) {
    return path;
  }
  if (path.startsWith("/")) {
    const siteRoot = SITE_ROOT.replace(/\/$/, "");
    return `${siteRoot}${path}`;
  }
  return `${SITE_ROOT}${path}`;
}

function setStatus(message, tone = "secondary") {
  statusBox.textContent = message;
  statusBox.className = `alert alert-${tone} mt-3 mb-0 py-2`;
}

function showProgress(fraction) {
  progressWrap.hidden = false;
  progressBar.style.width = `${Math.round(fraction * 100)}%`;
}

function hideProgress() {
  progressWrap.hidden = true;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load ${url} (${response.status})`);
  }
  return response.json();
}

// Pre-start: only the small manifest, so the intro can describe the actual
// model and download size before the user opts in to the heavy fetches.
async function loadManifest() {
  manifest = await fetchJson(`${DATA_BASE}manifest.json`);
  if (manifest.version !== 2) {
    throw new Error(`unsupported manifest version ${manifest.version}; rebuild with \`rake build_embeddings\``);
  }

  // Guard: the page's model registry must agree with what the data was
  // built with. A mismatch means data and page code are out of sync.
  const spec = getModelSpec(manifest.model.key);
  if (spec.hf_id !== manifest.model.hf_id || spec.dim !== manifest.model.dim) {
    throw new Error(
      `collection was built with ${manifest.model.hf_id} (${manifest.model.dim}-dim) but this page expects ` +
        `${spec.hf_id} (${spec.dim}-dim) — rebuild with \`rake build_embeddings\``,
    );
  }
  return spec;
}

async function loadArtifacts() {
  const indexJson = await fetchJson(`${DATA_BASE}index.json`);
  indexByFilename = new Map(indexJson.items.map((item) => [item.filename, item]));

  const binResponse = await fetch(`${DATA_BASE}${manifest.embeddings.file}`);
  if (!binResponse.ok) {
    throw new Error(`failed to load ${manifest.embeddings.file} (${binResponse.status})`);
  }
  blob = new Int8Array(await binResponse.arrayBuffer());

  const expected = manifest.embeddings.count * manifest.model.dim;
  if (blob.length !== expected) {
    throw new Error(`embeddings.bin has ${blob.length} values, expected ${expected} — rebuild the data`);
  }
}

async function loadModel(devicePreference = "auto") {
  const perFile = new Map();
  const progressCallback = (report) => {
    if (report.status === "progress" && report.total) {
      perFile.set(report.file, report.loaded / report.total);
      let sum = 0;
      perFile.forEach((value) => {
        sum += value;
      });
      showProgress(sum / perFile.size);
    }
  };

  // WebGPU when a usable adapter exists, WASM otherwise. Probe the adapter
  // up front: a failed session creation poisons backend state, so catching
  // afterwards is not a reliable fallback. Weights are always the same
  // quantized files the build used, so embeddings stay comparable.
  const options = { progress_callback: progressCallback, device: "wasm" };
  if (devicePreference === "webgpu" && navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (adapter) {
      options.device = "webgpu";
    }
  }
  embedder = await loadEmbedder(transformers, manifest.model.key, options);
  activeDevice = options.device;
}

function scoreQueryVector(queryVector) {
  const scores = scoreAll(queryVector, blob, manifest.embeddings.count, manifest.model.dim);
  const matches = topK(scores, manifest.top_k).map((row) => ({
    filename: manifest.filenames[row],
    score: scores[row],
    percent: calibratedPercent(scores[row], manifest.score_calibration),
  }));
  return { scores, matches };
}

function hasDegenerateScores(scores) {
  let min = Infinity;
  let max = -Infinity;
  for (const score of scores) {
    if (!Number.isFinite(score)) {
      return true;
    }
    min = Math.min(min, score);
    max = Math.max(max, score);
  }
  return max - min < 1e-6;
}

// Self-check: re-embed the first indexed image and require near-perfect
// similarity against its own stored vector (row 0 of the blob). Returns
// true when the backend looks healthy, false when it should be replaced.
// A seed image that cannot be fetched skips the check rather than failing
// startup — that is a network problem, not a backend problem.
async function verifyBackend() {
  const seedItem = indexByFilename.get(manifest.filenames[0]);
  const seedImagePath = resolveAssetUrl(seedItem?.image_path);
  if (!seedImagePath) {
    return true;
  }

  let seedBlob;
  try {
    const response = await fetch(seedImagePath);
    if (!response.ok) {
      return true;
    }
    seedBlob = await response.blob();
  } catch {
    return true;
  }

  try {
    const image = await transformers.RawImage.fromBlob(seedBlob);
    const queryVector = await embedImage(embedder, image, manifest.preprocessing);
    const { scores } = scoreQueryVector(queryVector);
    if (hasDegenerateScores(scores)) {
      return false;
    }
    return scores[0] >= SELF_CHECK_MIN_SCORE;
  } catch {
    return false;
  }
}

async function disposeEmbedder() {
  try {
    await embedder?.model?.dispose();
  } catch {
    // Best-effort: a backend broken enough to fail the self-check may also
    // fail to release its session cleanly.
  }
  embedder = null;
}

async function ensureWorkingBackend() {
  let ok = false;
  try {
    await loadModel("webgpu");
    ok = await verifyBackend();
  } catch {
    // Session creation can fail outright on some driver stacks; a WASM
    // attempt may still succeed (and if backend state is truly poisoned,
    // it will fail below and surface its own error).
    ok = false;
  }
  if (ok) {
    return;
  }

  setStatus("WebGPU backend produced unstable results, switching to WASM ...", "warning");
  await disposeEmbedder();
  await loadModel("wasm");
  const wasmOk = await verifyBackend();
  if (!wasmOk) {
    throw new Error("model backend self-check failed; search results may be unreliable");
  }
}

function renderResults(matches, elapsedMs) {
  resultsGrid.replaceChildren();
  resultsMeta.textContent = `Top ${matches.length} of ${manifest.embeddings.count} images in ${elapsedMs.toFixed(0)} ms.`;

  for (const match of matches) {
    const item = indexByFilename.get(match.filename) || {
      filename: match.filename,
      image_path: `objects/${match.filename}`,
    };
    const itemUrl = resolveItemUrl(item);

    const col = document.createElement("div");
    col.className = "col";

    const card = document.createElement("article");
    card.className = "card h-100";

    const img = document.createElement("img");
    img.src = resolveAssetUrl(item.thumb_path || item.image_path);
    img.alt = item.title || item.filename;
    img.loading = "lazy";
    img.className = "card-img-top object-fit-contain bg-body-tertiary";
    img.style.height = "140px";

    const body = document.createElement("div");
    body.className = "card-body p-2";

    const badge = document.createElement("span");
    badge.className = "badge text-bg-primary mb-1";
    badge.textContent = `${match.percent.toFixed(0)}% match`;

    const title = document.createElement("h3");
    title.className = "card-title fs-6 mb-1";
    if (itemUrl) {
      const titleLink = document.createElement("a");
      titleLink.href = itemUrl;
      titleLink.className = "link-dark text-decoration-none";
      titleLink.textContent = item.title || item.filename;
      title.appendChild(titleLink);
    } else {
      title.textContent = item.title || item.filename;
    }

    body.append(badge, title);

    const links = document.createElement("div");
    links.className = "d-flex flex-wrap gap-2 small";

    if (itemUrl) {
      const itemLink = document.createElement("a");
      itemLink.href = itemUrl;
      itemLink.className = "link-primary";
      itemLink.textContent = "View item";
      links.appendChild(itemLink);
    }

    if (links.childElementCount > 0) {
      body.appendChild(links);
    }

    card.append(img, body);
    col.appendChild(card);
    resultsGrid.appendChild(col);
  }
}

async function runSearch() {
  if (!selectedFile) {
    setStatus("Select an image before searching.", "warning");
    return;
  }

  searchButton.disabled = true;
  setStatus("Computing query embedding ...", "info");
  const start = performance.now();

  try {
    const image = await transformers.RawImage.fromBlob(selectedFile);
    const queryVector = await embedImage(embedder, image, manifest.preprocessing);

    const { matches } = scoreQueryVector(queryVector);

    renderResults(matches, performance.now() - start);
    setStatus("Done. Select another image to search again.", "success");
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, "danger");
    resultsMeta.textContent = "Unable to compute results.";
    resultsGrid.replaceChildren();
  } finally {
    searchButton.disabled = false;
  }
}

function handleFileSelection(event) {
  const [file] = event.target.files || [];
  selectedFile = file || null;

  if (!selectedFile) {
    previewWrap.hidden = true;
    previewImage.removeAttribute("src");
    searchButton.disabled = true;
    setStatus("No image selected.");
    return;
  }

  const previewUrl = URL.createObjectURL(selectedFile);
  previewImage.onload = () => URL.revokeObjectURL(previewUrl);
  previewImage.src = previewUrl;
  previewWrap.hidden = false;
  resultsGrid.replaceChildren();
  resultsMeta.textContent = "Image ready.";
  setStatus("Ready to search.", "secondary");
  searchButton.disabled = false;
}

// Post-start: fetch the collection index + embeddings and download the model.
// On failure the intro card returns with the error and an enabled start
// button, so a transient problem (network blip, CDN hiccup) is retryable
// without reloading the page.
async function startSearch() {
  startButton.disabled = true;
  introCard.hidden = true;
  searchUi.hidden = false;
  resultsSection.hidden = false;

  try {
    setStatus("Loading collection data ...", "secondary");
    await loadArtifacts();

    const spec = getModelSpec(manifest.model.key);
    setStatus(`Loading ${spec.label} model (cached by your browser after the first visit) ...`, "secondary");
    await ensureWorkingBackend();
    hideProgress();

    setStatus(
      `Ready. ${manifest.embeddings.count} images indexed with ${spec.label}, "${manifest.preprocessing}" preprocessing (backend: ${activeDevice}).`,
      "success",
    );
    imageInput.disabled = false;
  } catch (error) {
    hideProgress();
    searchUi.hidden = true;
    resultsSection.hidden = true;
    introCard.hidden = false;
    introStatus.textContent = `Startup failed: ${error.message} — you can try again.`;
    introStatus.hidden = false;
    startButton.disabled = false;
  }
}

async function init() {
  try {
    const spec = await loadManifest();

    const pageLibraryVersion = transformers.env?.version;
    const buildLibraryVersion = manifest.library?.version;
    if (pageLibraryVersion && buildLibraryVersion && pageLibraryVersion !== buildLibraryVersion) {
      console.warn(
        `embeddings: data was built with @huggingface/transformers ${buildLibraryVersion} but this page loads ` +
          `${pageLibraryVersion} — update the CDN pin in app.js or re-run \`rake build_embeddings\` to keep ` +
          `build and query preprocessing identical`,
      );
    }

    downloadNote.textContent =
      `Starting the search downloads the ${spec.label} image-analysis model ` +
      `(about ${spec.approx_download_mb} MB) and the collection index for ` +
      `${manifest.embeddings.count} images to your browser. This happens once; ` +
      `your browser caches the files for future visits.`;
    startButton.disabled = false;
    startButton.addEventListener("click", startSearch);
    imageInput.addEventListener("change", handleFileSelection);
    searchButton.addEventListener("click", runSearch);
  } catch (error) {
    introStatus.textContent = `This search is unavailable: ${error.message}`;
    introStatus.hidden = false;
  }
}

init();
