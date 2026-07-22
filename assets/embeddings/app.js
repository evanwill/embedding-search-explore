/**
 * Embedding search page: reverse image lookup + text-to-image search.
 *
 * Configures itself entirely from data/manifest.json (written by the build):
 * which model to load, which preprocessing profile to apply, quantization
 * parameters, score calibration, and whether text search is available.
 * Rebuilding with different settings in embeddings/config-embeddings.yml
 * updates this page with no code edits.
 *
 * The two model towers load independently so a visitor only downloads what
 * they use: the vision tower (~85 MB for CLIP) for image queries, the text
 * tower (~65 MB) for free-text queries. The intro card offers a start button
 * per mode, and the mode toggle lazy-loads the other tower on request.
 */

import * as transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import {
  getModelSpec,
  loadEmbedder,
  embedImage,
  loadTextEmbedder,
  embedText,
  scoreAll,
  topK,
  calibratedPercent,
  QUANT_SCALE,
} from "./embedding-core.mjs";

const appRoot = document.getElementById("reverse-lookup-app");
const DATA_BASE = normalizeBase(appRoot?.dataset.dataBase || "assets/embeddings/data/");
const SITE_ROOT = normalizeSiteRoot(appRoot?.dataset.siteRoot || "/");

const introCard = document.getElementById("rl-intro");
const introStatus = document.getElementById("rl-intro-status");
const downloadNote = document.getElementById("rl-download-note");
const startButton = document.getElementById("rl-start-button");
const startTextButton = document.getElementById("rl-start-text-button");
const searchUi = document.getElementById("rl-search-ui");
const resultsSection = document.getElementById("rl-results-section");
const modeImageButton = document.getElementById("rl-mode-image");
const modeTextButton = document.getElementById("rl-mode-text");
const imagePanel = document.getElementById("rl-image-panel");
const imageEnable = document.getElementById("rl-image-enable");
const imageEnableNote = document.getElementById("rl-image-enable-note");
const imageEnableButton = document.getElementById("rl-image-enable-button");
const imageControls = document.getElementById("rl-image-controls");
const textPanel = document.getElementById("rl-text-panel");
const textEnable = document.getElementById("rl-text-enable");
const textEnableNote = document.getElementById("rl-text-enable-note");
const textEnableButton = document.getElementById("rl-text-enable-button");
const textControls = document.getElementById("rl-text-controls");
const imageInput = document.getElementById("rl-image-input");
const searchButton = document.getElementById("rl-search-button");
const textInput = document.getElementById("rl-text-input");
const textSearchButton = document.getElementById("rl-text-search-button");
const statusBox = document.getElementById("rl-status");
const progressWrap = document.getElementById("rl-progress-wrap");
const progressBar = document.getElementById("rl-progress");
const previewWrap = document.getElementById("rl-preview-wrap");
const previewImage = document.getElementById("rl-preview");
const resultsMeta = document.getElementById("rl-results-meta");
const resultsGrid = document.getElementById("rl-results");

let manifest = null;
let modelSpec = null;
let blob = null;
let indexByFilename = new Map();
let artifactsLoaded = false;
let imageEmbedder = null;
let textEmbedder = null;
let imageDevice = null;
let textDevice = null;
let selectedFile = null;
let currentMode = "image";

// Minimum acceptable self-similarity for the backend self-checks: the image
// check re-embeds a known indexed image against its own stored vector; the
// text check re-embeds the manifest's reference probe string against its
// build-time vector. A working backend lands near 1.0 (int8 quantization
// costs a little); a broken one produces garbage vectors that score far
// lower. A score threshold is robust where a rank-1 check is not:
// collections with near-duplicate images can legitimately rank a sibling
// above the seed within numerical noise.
const SELF_CHECK_MIN_SCORE = 0.95;

// Text-image cosine scores live far below the image-image range the
// manifest's calibration was computed from, so text results are displayed
// with a per-query calibration instead: floor = this percentile of the
// query's scores across the collection, ceiling = the query's best score.
const TEXT_CALIBRATION_FLOOR_PERCENTILE = 0.05;

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

function textSearchAvailable() {
  return Boolean(manifest?.text_search?.available);
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

function makeProgressCallback() {
  const perFile = new Map();
  return (report) => {
    if (report.status === "progress" && report.total) {
      perFile.set(report.file, report.loaded / report.total);
      let sum = 0;
      perFile.forEach((value) => {
        sum += value;
      });
      showProgress(sum / perFile.size);
    }
  };
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load ${url} (${response.status})`);
  }
  return response.json();
}

// Pre-start: only the small manifest, so the intro can describe the actual
// model and download sizes before the user opts in to the heavy fetches.
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
  modelSpec = spec;
  return spec;
}

async function loadArtifacts() {
  if (artifactsLoaded) {
    return;
  }
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
  artifactsLoaded = true;
}

// WebGPU when a usable adapter exists, WASM otherwise. Probe the adapter
// up front: a failed session creation poisons backend state, so catching
// afterwards is not a reliable fallback.
async function resolveDevice(preference) {
  if (preference === "webgpu" && navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (adapter) {
      return "webgpu";
    }
  }
  return "wasm";
}

// Weights are always the same quantized files the build used, so embeddings
// stay comparable across devices.
async function loadImageModel(devicePreference) {
  const device = await resolveDevice(devicePreference);
  imageEmbedder = await loadEmbedder(transformers, manifest.model.key, {
    progress_callback: makeProgressCallback(),
    device,
  });
  imageDevice = device;
}

async function loadTextModel(devicePreference) {
  const device = await resolveDevice(devicePreference);
  textEmbedder = await loadTextEmbedder(transformers, manifest.model.key, {
    progress_callback: makeProgressCallback(),
    device,
  });
  textDevice = device;
}

function computeScores(queryVector) {
  return scoreAll(queryVector, blob, manifest.embeddings.count, manifest.model.dim);
}

function rankMatches(scores, calibration) {
  return topK(scores, manifest.top_k).map((row) => ({
    filename: manifest.filenames[row],
    score: scores[row],
    percent: calibratedPercent(scores[row], calibration),
  }));
}

function perQueryCalibration(scores) {
  const sorted = Float32Array.from(scores).sort();
  const floor = sorted[Math.floor(sorted.length * TEXT_CALIBRATION_FLOOR_PERCENTILE)];
  const ceiling = sorted[sorted.length - 1];
  return { floor, ceiling };
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

// Image self-check: re-embed the first indexed image and require near-perfect
// similarity against its own stored vector (row 0 of the blob). Returns
// true when the backend looks healthy, false when it should be replaced.
// A seed image that cannot be fetched skips the check rather than failing
// startup — that is a network problem, not a backend problem.
async function verifyImageBackend() {
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
    const queryVector = await embedImage(imageEmbedder, image, manifest.preprocessing);
    const scores = computeScores(queryVector);
    if (hasDegenerateScores(scores)) {
      return false;
    }
    return scores[0] >= SELF_CHECK_MIN_SCORE;
  } catch {
    return false;
  }
}

// Text self-check: re-embed the manifest's reference probe string and require
// near-perfect similarity against its build-time vector. Data without a
// reference vector skips the check.
async function verifyTextBackend() {
  const reference = manifest.text_search?.reference;
  const scale = manifest.embeddings.scale || QUANT_SCALE;
  if (!reference?.vector || reference.vector.length !== manifest.model.dim) {
    return true;
  }

  try {
    const queryVector = await embedText(textEmbedder, reference.text);
    let dot = 0;
    for (let i = 0; i < queryVector.length; i += 1) {
      dot += queryVector[i] * (reference.vector[i] / scale);
    }
    if (!Number.isFinite(dot)) {
      return false;
    }
    return dot >= SELF_CHECK_MIN_SCORE;
  } catch {
    return false;
  }
}

async function disposeImageEmbedder() {
  try {
    await imageEmbedder?.model?.dispose();
  } catch {
    // Best-effort: a backend broken enough to fail the self-check may also
    // fail to release its session cleanly.
  }
  imageEmbedder = null;
  imageDevice = null;
}

async function disposeTextEmbedder() {
  try {
    await textEmbedder?.textModel?.dispose();
  } catch {
    // Best-effort, as above.
  }
  textEmbedder = null;
  textDevice = null;
}

// Load a tower with WebGPU when available, verify it against known data, and
// fall back to WASM (re-verifying) when the self-check fails. If the other
// tower already fell back to WASM, skip WebGPU — the driver stack has shown
// itself untrustworthy once already.
async function ensureImageBackend() {
  if (imageEmbedder) {
    return;
  }
  const preference = textDevice === "wasm" ? "wasm" : "webgpu";
  let ok = false;
  try {
    await loadImageModel(preference);
    ok = await verifyImageBackend();
  } catch {
    // Session creation can fail outright on some driver stacks; a WASM
    // attempt may still succeed (and if backend state is truly poisoned,
    // it will fail below and surface its own error).
    ok = false;
  }
  if (ok) {
    return;
  }

  setStatus("The compute backend produced unstable results, switching to WASM ...", "warning");
  await disposeImageEmbedder();
  await loadImageModel("wasm");
  const wasmOk = await verifyImageBackend();
  if (!wasmOk) {
    await disposeImageEmbedder();
    throw new Error("image model backend self-check failed; search results may be unreliable");
  }
}

async function ensureTextBackend() {
  if (textEmbedder) {
    return;
  }
  const preference = imageDevice === "wasm" ? "wasm" : "webgpu";
  let ok = false;
  try {
    await loadTextModel(preference);
    ok = await verifyTextBackend();
  } catch {
    ok = false;
  }
  if (ok) {
    return;
  }

  setStatus("The compute backend produced unstable results, switching to WASM ...", "warning");
  await disposeTextEmbedder();
  await loadTextModel("wasm");
  const wasmOk = await verifyTextBackend();
  if (!wasmOk) {
    await disposeTextEmbedder();
    throw new Error("text model backend self-check failed; search results may be unreliable");
  }
}

function updateModeUi() {
  modeImageButton.classList.toggle("active", currentMode === "image");
  modeTextButton.classList.toggle("active", currentMode === "text");
  imagePanel.hidden = currentMode !== "image";
  textPanel.hidden = currentMode !== "text";
  imageEnable.hidden = Boolean(imageEmbedder);
  imageControls.hidden = !imageEmbedder;
  textEnable.hidden = Boolean(textEmbedder);
  textControls.hidden = !textEmbedder;
}

function setReadyStatus() {
  const count = manifest.embeddings.count;
  if (currentMode === "text") {
    setStatus(
      `Ready for text search. ${count} images indexed with ${modelSpec.label} (backend: ${textDevice}).`,
      "success",
    );
  } else {
    setStatus(
      `Ready for image search. ${count} images indexed with ${modelSpec.label}, "${manifest.preprocessing}" preprocessing (backend: ${imageDevice}).`,
      "success",
    );
  }
}

function setMode(mode) {
  currentMode = mode;
  updateModeUi();
  const loaded = mode === "text" ? Boolean(textEmbedder) : Boolean(imageEmbedder);
  if (loaded) {
    setReadyStatus();
  }
}

function renderResults(matches, elapsedMs, badgeLabel = "match") {
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
    badge.textContent = `${match.percent.toFixed(0)}% ${badgeLabel}`;

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

async function runImageSearch() {
  if (!selectedFile) {
    setStatus("Select an image before searching.", "warning");
    return;
  }

  searchButton.disabled = true;
  setStatus("Computing query embedding ...", "info");
  const start = performance.now();

  try {
    const image = await transformers.RawImage.fromBlob(selectedFile);
    const queryVector = await embedImage(imageEmbedder, image, manifest.preprocessing);
    const matches = rankMatches(computeScores(queryVector), manifest.score_calibration);

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

async function runTextSearch() {
  const query = textInput.value.trim();
  if (!query) {
    setStatus("Type a few words describing what you are looking for.", "warning");
    return;
  }

  textSearchButton.disabled = true;
  setStatus("Computing text embedding ...", "info");
  const start = performance.now();

  try {
    const queryVector = await embedText(textEmbedder, query);
    const scores = computeScores(queryVector);
    // "Relative match": how much better than the field, not the same scale
    // as image-mode percentages.
    const matches = rankMatches(scores, perQueryCalibration(scores));

    renderResults(matches, performance.now() - start, "relative match");
    setStatus(`Done. Results for "${query}" — edit the description to search again.`, "success");
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, "danger");
    resultsMeta.textContent = "Unable to compute results.";
    resultsGrid.replaceChildren();
  } finally {
    textSearchButton.disabled = false;
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

// Post-start: fetch the collection index + embeddings once, then download
// only the tower for the mode the visitor chose. On failure the intro card
// returns with the error and enabled start buttons, so a transient problem
// (network blip, CDN hiccup) is retryable without reloading the page.
async function startMode(mode) {
  startButton.disabled = true;
  startTextButton.disabled = true;
  introCard.hidden = true;
  searchUi.hidden = false;
  resultsSection.hidden = false;
  currentMode = mode;
  updateModeUi();

  try {
    setStatus("Loading collection data ...", "secondary");
    await loadArtifacts();

    if (mode === "text") {
      setStatus(`Loading ${modelSpec.label} text model (cached by your browser after the first visit) ...`, "secondary");
      await ensureTextBackend();
      textInput.disabled = false;
      textSearchButton.disabled = false;
    } else {
      setStatus(`Loading ${modelSpec.label} image model (cached by your browser after the first visit) ...`, "secondary");
      await ensureImageBackend();
      imageInput.disabled = false;
    }
    hideProgress();
    updateModeUi();
    setReadyStatus();
  } catch (error) {
    hideProgress();
    searchUi.hidden = true;
    resultsSection.hidden = true;
    introCard.hidden = false;
    introStatus.textContent = `Startup failed: ${error.message} — you can try again.`;
    introStatus.hidden = false;
    startButton.disabled = false;
    startTextButton.disabled = !textSearchAvailable();
  }
}

// Lazy-enable the tower the visitor did not start with, from inside its
// mode panel, with its own consent button.
async function enableImageMode() {
  imageEnableButton.disabled = true;
  try {
    setStatus(`Loading ${modelSpec.label} image model (cached by your browser after the first visit) ...`, "secondary");
    await ensureImageBackend();
    hideProgress();
    imageInput.disabled = false;
    updateModeUi();
    setReadyStatus();
  } catch (error) {
    hideProgress();
    setStatus(`Could not enable image search: ${error.message} — you can try again.`, "danger");
    imageEnableButton.disabled = false;
  }
}

async function enableTextMode() {
  textEnableButton.disabled = true;
  try {
    setStatus(`Loading ${modelSpec.label} text model (cached by your browser after the first visit) ...`, "secondary");
    await ensureTextBackend();
    hideProgress();
    textInput.disabled = false;
    textSearchButton.disabled = false;
    updateModeUi();
    setReadyStatus();
  } catch (error) {
    hideProgress();
    setStatus(`Could not enable text search: ${error.message} — you can try again.`, "danger");
    textEnableButton.disabled = false;
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

    const textAvailable = textSearchAvailable();
    const textMb = manifest.text_search?.approx_download_mb;

    let note =
      `Starting the image search downloads the ${spec.label} image-analysis model ` +
      `(about ${spec.approx_download_mb} MB) and the collection index for ` +
      `${manifest.embeddings.count} images to your browser.`;
    if (textAvailable) {
      note +=
        ` Starting the text search instead downloads a smaller language model ` +
        `(about ${textMb} MB) — only what you choose is downloaded.`;
    }
    note += ` Downloads happen once; your browser caches the files for future visits.`;
    downloadNote.textContent = note;

    imageEnableNote.textContent =
      `Image search needs the ${spec.label} image-analysis model ` +
      `(about ${spec.approx_download_mb} MB, downloaded once and cached by your browser).`;
    textEnableNote.textContent =
      `Text search needs the ${spec.label} language model ` +
      `(about ${textMb} MB, downloaded once and cached by your browser).`;

    if (textAvailable) {
      startTextButton.hidden = false;
      startTextButton.disabled = false;
      modeTextButton.hidden = false;
    }

    startButton.disabled = false;
    startButton.addEventListener("click", () => startMode("image"));
    startTextButton.addEventListener("click", () => startMode("text"));
    modeImageButton.addEventListener("click", () => setMode("image"));
    modeTextButton.addEventListener("click", () => setMode("text"));
    imageEnableButton.addEventListener("click", enableImageMode);
    textEnableButton.addEventListener("click", enableTextMode);
    imageInput.addEventListener("change", handleFileSelection);
    searchButton.addEventListener("click", runImageSearch);
    textSearchButton.addEventListener("click", runTextSearch);
    textInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !textSearchButton.disabled) {
        runTextSearch();
      }
    });
  } catch (error) {
    introStatus.textContent = `This search is unavailable: ${error.message}`;
    introStatus.hidden = false;
  }
}

init();
