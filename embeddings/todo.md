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
