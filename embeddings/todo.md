# migrate embeddings to do

- check all filepaths to ensure they match new locations
- update configuration options
    - read embeddings specific settings from file "embeddings/config-embeddings.yml" (rather than old "config.yml")
    - read metadata csv from central project configuration, `metadata` key in "_config.yml"
        - this metadata may contain many item types and extra information.
        - filter out to only image items ("display_template" = `image`).
        - image file location is listed in "object_location" field.
- update "pages/embeddings.html" and "assets/embeddings/app.js" to refine search for new context in the collection website. 
    - update styles to work with template collectionbuilder setup
    - update results list to provide link to Item page in the collection site (if item has a parentid value, /items/ + parentid + .html + # + objectid, else /items/ + objectid + .html)

## implementation plan

### 1. align config sources and path handling
- status: completed (2026-07-21)
- inventory every path and config read in `rakelib/build_embeddings.rake`, `embeddings/scripts/build_embeddings.mjs`, `assets/embeddings/app.js`, and `pages/embeddings.html`.
- replace any remaining assumptions about root-level `config.yml` with:
    - embeddings settings from `embeddings/config-embeddings.yml`.
    - collection metadata dataset name from `_config.yml` key `metadata` (no `.csv` extension).
- centralize resolved paths in the build scripts (source metadata CSV, objects directory, output `assets/embeddings/data/`) so one path strategy is used everywhere.
- ensure generated `manifest.json` records only values needed by the browser app and that all data URLs are relative to the embeddings page.

### 2. update metadata ingestion for CollectionBuilder semantics
- status: completed (2026-07-21)
- load active metadata from `_data/<metadata>.csv` using the `_config.yml` `metadata` pointer.
- normalize row values for robust filtering (trim whitespace and handle case variants where needed).
- filter rows to image items only: `display_template == image`.
- read image path from `object_location` and resolve to actual source files in the collection.
- keep stable item identifiers (`objectid`, `parentid`) available through the indexing pipeline for later link generation.
- fail fast with clear build errors when required fields are missing (`objectid`, `display_template`, `object_location`), and warn on skipped rows/files.

### 3. revise index payload and item-link logic
- status: completed (2026-07-21)
- adjust `indexRecord()` in `embeddings/scripts/build_embeddings.mjs` so `index.json` includes only fields needed for result display plus linking keys.
- include computed `item_url` in each indexed record using CollectionBuilder item URL rules:
    - with `parentid`: `/items/<parentid>.html#<objectid>`
    - without `parentid`: `/items/<objectid>.html`
- keep payload compact by trimming unused metadata fields and preserving current lightweight data goals.

### 4. adapt browser app and page UI
- status: completed (2026-07-21)
- update `assets/embeddings/app.js` to consume any new manifest/index field names and render result cards with links to item pages.
- ensure link rendering handles absent/empty `parentid` safely and never outputs broken anchors.
- refresh `pages/embeddings.html` markup and classes to align with existing CollectionBuilder/Bootstrap 5 patterns while preserving progressive disclosure and consent-to-load model behavior.
- verify the page works as a normal CollectionBuilder page route and that script/data paths remain valid after Jekyll build.

### 5. polish, documentation, and verification
- status: completed (2026-07-21)
- run `rake build_embeddings` and verify regenerated outputs in `assets/embeddings/data/` (`manifest.json`, `embeddings.bin`, `index.json`, `build-info.json`).
- run `bundle exec jekyll build` (and optionally `bundle exec jekyll serve`) to validate the page in site context.
- manually test search flow:
    - upload image, compute query embedding, return ranked results.
    - confirm each result links correctly for both parent and non-parent items.
    - confirm behavior when non-image records exist in metadata.
- update `embeddings/README.md` with final migration notes (config source expectations, required metadata columns, item URL behavior).
- if needed, add a short troubleshooting subsection for common migration failures (missing metadata key, bad object paths, missing object files).

### 6. optional hardening tasks (if time remains)
- add a small validation task or script to check metadata prerequisites before full embedding generation.
- add a tiny fixture-based smoke test for URL generation logic (parent vs non-parent) to prevent regressions.
- add a build summary line that reports total metadata rows, image rows kept, rows skipped, and embeddings written.

--------

# text-to-image search prototype

Add free-text search ("miners with candles", "wooden support timbers") against the existing image embeddings. Because the collection is already embedded with CLIP, the image side needs **no rebuild**: CLIP's text tower projects text into the same 512-dim space as `image_embeds`, so a text query scores against the existing `embeddings.bin` with the exact `scoreAll`/`topK` path already in use.

Key facts constraining the design:

- Only `clip` has a usable text tower in our registry (`mobilenet`/MobileCLIP has one too, but out of scope for the prototype; `dinov2` has none). Text search must be feature-gated on the model.
- The q8 text tower (`text_model_quantized.onnx`) is **64.5 MB** on the HF hub, plus a small tokenizer. It must be lazy-loaded on first text search, not bundled into the image-search start flow.
- Image-image cosine scores (current calibration floor 0.506) and text-image cosine scores live in very different ranges — CLIP text-image similarities typically land around 0.15–0.35. The existing `score_calibration` **cannot** be used to display text-result percentages; text mode needs its own display scaling.
- Text queries have no build-time counterpart, so the strict build/browser parity contract doesn't apply to the text tower — but preprocessing still matters indirectly: image embeddings from the `standard` profile sit in CLIP's natural space (good for text queries); `binary`/`lineart` embeddings are shifted, so text search quality degrades on those profiles. Worth a documented warning, not a hard block.

### 1. shared core: text embedding support (`assets/embeddings/embedding-core.mjs`)
- status: completed (2026-07-21)
- add `supports_text: true` to the `clip` entry in `MODEL_REGISTRY` (plus `text_approx_download_mb: 65`); leave other models without it.
- add `loadTextEmbedder(transformers, key, options)`: `AutoTokenizer.from_pretrained` + `CLIPTextModelWithProjection.from_pretrained` with the registry's `hf_id` and `dtype: "q8"`, returning `{ tokenizer, textModel, spec }`. Throw a clear error for models without `supports_text`.
- add `embedText(textEmbedder, query)`: tokenize with `padding: true, truncation: true`, run the model, take `text_embeds`, L2-normalize, return `Float32Array` — mirroring `embedImage` so the module stays environment-neutral (a Node eval script can reuse it unchanged).

### 2. build/manifest: feature flag export + text reference vector
- status: completed (2026-07-21)
- `embeddings/scripts/build_embeddings.mjs`: write a `text_search` block into the manifest — `{ available: <model supports_text>, approx_download_mb, reference }`. The block tells the page whether to offer text search without hardcoding model knowledge page-side.
- **build-time text reference vector**: when text search is available, the build loads the text tower once, embeds a fixed probe string, and stores its int8-quantized vector in `text_search.reference`. This gives the browser's text backend the same verified-fallback treatment as the image backend — essential now that a text-only visitor never loads the vision tower the image self-check depends on. (~2 KB of manifest JSON; the text tower download is cached on the build machine.)
- update `embeddings/scripts/schemas/manifest.schema.json` for the new optional block (manifest stays version 2 — additive change).
- config knob in `embeddings/config-embeddings.yml`: `text_search: true|false` (default true) to let a collection turn the feature off even on clip.

### 3. page UI (`pages/embeddings.html`)
- status: completed (2026-07-21)
- **dual-choice intro**: before anything heavy loads, the intro card offers two start buttons — "Start image search" (~85 MB vision tower) and "Start text search" (~65 MB text tower) — each with its own download size in the note, so the page only downloads the tower the visitor actually wants. The text button only renders when `manifest.text_search.available`; on other models the page looks exactly as it does today.
- after starting, a mode switch (Bootstrap 5 `nav nav-pills`) sits above the search controls: "Search by image" / "Search by text". Switching to a mode whose tower isn't loaded yet shows a small enable card with that tower's download note and an explicit enable button — the other tower is never downloaded without consent.
- text panel: text input (`maxlength` ~200) + search button.
- reuse the existing status box, progress bar, and results grid — results render identically (same cards, same item links) regardless of query mode.

### 4. browser app (`assets/embeddings/app.js`)
- status: completed (2026-07-21)
- restructure startup: collection artifacts (index + embeddings.bin) load once on either start path; each tower loads independently on demand (`ensureImageBackend` / `ensureTextBackend`), each with its own WebGPU-verify-fallback-to-WASM cycle and its own active-device tracking.
- text backend self-check: embed the manifest's reference probe string and require cosine ≥ 0.95 against the stored build-time vector (plus the degenerate-scores guard); on failure dispose and retry on WASM, mirroring the image path. If the image tower already fell back to WASM, start the text tower on WASM directly.
- text query flow: `embedText` → `scoreAll` against the existing blob → `topK` → render.
- **text-mode score display**: per-query calibration instead of the manifest's image-image calibration — floor = 5th percentile of this query's scores across the whole collection, ceiling = the query's max score. Self-contained, no build change, and gives a meaningful "how much better than the field is this result" percentage. Label the badge differently in text mode (e.g. "relative match") so the numbers aren't read as the same scale as image mode.
- guard rails: empty-query no-op; disable the text search button while embedding; tower load failure shows a retryable error without breaking the other mode.

### 5. verification + docs
- status: mostly completed (2026-07-21) — browser test remains
- [done] Node smoke script (`scripts/text_search_smoke.mjs`, kept as a permanent utility): runs captions against `embeddings.bin` and prints top-5 titles per query. Validated on the collection — "mine tunnel interior" → mine-level photos (scores ~0.31), "wooden support timbers" → lumber camps, "men standing in a group" → group portraits; "ore cart on rails" was the weakest (generic wheeled vehicles). Scores landed in the 0.23–0.31 band, confirming the per-query calibration decision.
- [done] text backend parity verified in Node: re-embedding the manifest reference string scores 1.003 against the stored build-time vector (threshold 0.95).
- [pending] browser test: dual start buttons appear, text-only start downloads only the ~65 MB text tower, results render with links, mode toggle asks consent before the other tower's download; `dinov2` build hides the text option (rebuild with dinov2 briefly or hand-edit a manifest copy to verify the gate).
- [done] docs: README gained a "text-to-image search" feature section; get-started gained the `text_search` config row and a text-search smoke-test step.

### 6. later / out of scope for the prototype
- prompt templating ("a photo of {query}") — CLIP often retrieves better with template prompts; expose as a tuning knob after evaluating raw queries on real collections.
- MobileCLIP text tower (much smaller download) once its text side is validated in Transformers.js.
- text-query eval harness: a labeled `query,expected_filename` CSV scored by rank, alongside the existing perturbation harness.
- hybrid search (text + example image combined scoring).

## Future ideas

- **Zero-install build page**: since the model already runs in the browser, an admin-facing static page could accept a dragged-in folder of images + metadata CSV, compute all embeddings client-side, and download the `data/` artifacts — removing Ruby *and* Node from the workflow entirely ("open a page, drop a folder"). `embedding-core.mjs` is deliberately environment-neutral so such a page could reuse it unchanged.
- webgpu/wasm fallback test - implemented (2026-07-21): the build embeds a deterministic synthetic probe image (integer-math pixels, identical in Node and every browser) and stores its vector in `manifest.self_check.image_vector`; the browser regenerates the probe, re-embeds, and requires cosine ≥ 0.95 — fetch-free, decoder-free, and content-independent, symmetric with the text tower's probe-string check.

#### future progress: extending the eval harness with real test sets

The perturbation harness only measures same-domain robustness; the embossed-mark probe above shows real cross-domain performance can rank configurations very differently. As good test images accumulate (photos of physical marks whose collection counterpart is known), the harness should grow support for a labeled pair file — e.g. `docs/test/pairs.csv` with `query_file,target_filename` columns — and report rank/top-K hit rates over those real pairs alongside the synthetic perturbations. Model and preprocessing decisions should then weight the real-pair results over the synthetic ones. The one-off diagnostic that produced the cross-domain numbers above is `scripts/.diagnose-crossdomain.mjs` and can serve as the starting point.

------

# Testing

## evaluation results (printer's-marks prototype collection)

An evaluation harness (`scripts/eval_retrieval.mjs`) measures top-1/top-5 self-retrieval from perturbed queries so you can pick the configuration for your own collection empirically. The results below are from the 196-image printer's-marks collection this package was originally prototyped on (65 queries per perturbation; a "hit" requires the exact source file, so near-duplicate variants count as misses):

| model | profile | crop 10% | rotate 5° | photo sim |
|---|---|---|---|---|
| `clip` | `binary` | 92% / 100% | 83% / 91% | 95% / 98% |
| `clip` | `lineart` | 98% / 100% | 94% / 100% | 98% / 100% |
| `clip` | `standard` | 98% / 100% | 97% / 100% | 97% / 100% |
| `dinov2` | `binary` | 91% / 98% | 83% / 85% | 92% / 97% |
| `dinov2` | `lineart` | 97% / 100% | 98% / 100% | 98% / 100% |
| `dinov2` | `standard` | 97% / 100% | 98% / 100% | 98% / 100% |
| `mobilenet` | `lineart` | 9% / 22% | 6% / 17% | 6% / 15% |
| `mobilenet` | `standard` | 9% / 22% | 14% / 23% | 3% / 8% |

(cells are top-1 / top-5 hit rates)

**Cross-domain queries are the harder, more realistic test** — and synthetic perturbations miss it. A real-world probe (a photo of a blind-embossed mark, `docs/test/test.jpg`, whose inked line-art version is `docs/objects/10003.jpg`) reversed the picture the table above paints: with `lineart` preprocessing the target ranked #39 under `dinov2` and #15 under `clip`, while **`clip` + `binary` retrieved it at #1** (and `dinov2` + `binary` at #4). Binarization costs a few points of same-domain rotation robustness (thin binarized lines alias under rotation) but is decisively better when users photograph physical marks — hence the prototype's choice of `clip` + `binary` for that material.

**Caveat on `mobilenet`**: MobileCLIP-S0 retrieves exact uploads perfectly but is very sensitive to crops and rotations on this line-art collection — choose it only when download size is critical and queries will be close copies of collection images. Run the harness on your own material before trusting any of these numbers elsewhere.
