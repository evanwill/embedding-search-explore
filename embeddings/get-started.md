# get started

## prerequisites

- **Ruby + Rake** (the command entry point; already present if you work with Jekyll/CollectionBuilder)
- **Node.js 18+** (runs the embedding pipeline; install the LTS release from <https://nodejs.org/>)

Dependencies are installed with npm (bundled with Node) or pnpm — `rake setup_embeddings` finds whichever is available.

## quickstart

1. Review `embeddings/config-embeddings.yml` and pick a model and preprocessing profile for your material. Use `standard` preprocessing for photographs and continuous-tone images; use `binary` only for high-contrast material like printer's marks (see the README for trade-offs and evaluation results).

2. Make sure `_config.yml` points at your metadata CSV (`metadata: <name>`, no `.csv` extension) and that the CSV has the required columns: `objectid`, `display_template`, `object_location`. Only rows with `display_template` of `image` are embedded.

3. Install JavaScript dependencies:

```sh
rake setup_embeddings
```

4. Build the search data (the first run downloads model weights, which are cached for later runs):

```sh
rake build_embeddings
```

5. Serve the site and open the search page:

```sh
bundle exec jekyll serve
```

Then visit <http://localhost:4000/embeddings.html>.

## configuration

`embeddings/config-embeddings.yml` controls the pipeline:

| setting | values | notes |
|---|---|---|
| `model` | `clip`, `dinov2`, `mobilenet` | which embedding model to use; see README for trade-offs |
| `preprocessing` | `standard`, `lineart`, `binary` | image normalization applied identically at build time and to user uploads |
| `objects_dir` | path | collection images directory |
| `filename_field` | CSV column name | joins metadata rows to image files (defaults to `filename`) |
| `output_dir` | path | where artifacts are written (the search page expects `assets/embeddings/data/`) |
| `top_k` | integer | how many results the page shows |

The metadata CSV is not configured here — it is derived from the `metadata:` key in `_config.yml`, resolved as `_data/<metadata>.csv`.

The build exports the resolved configuration to `assets/embeddings/data/manifest.json`, and the search page configures itself from that manifest — after changing `config-embeddings.yml`, just re-run `rake build_embeddings`. No page edits are needed.

## generated artifacts

After `rake build_embeddings`, files are written to `assets/embeddings/data/`:

- `manifest.json` — build configuration export: model, preprocessing profile, quantization parameters, ordered filename list, score calibration
- `embeddings.bin` — collection vectors: `count × dim` int8 values (L2-normalized, scale 127), row-major; row *i* belongs to `manifest.filenames[i]`
- `index.json` — trimmed metadata records for the result cards, including `item_url` links to collection item pages
- `build-info.json` — counts, skipped images, unmatched metadata rows
- `preprocess.log` — plain-text build summary

Run `rake clean_data` to remove the generated artifacts.

## smoke test checklist

1. Run the build and check the summary line, e.g.:

```text
Summary: images_found=496, embeddings_built=496, skipped_images=0
```

Any skips are listed with per-file reasons in `build-info.json` and `preprocess.log`.

2. Launch the site (`bundle exec jekyll serve`) and open the search page. The intro card should name the model from your config and its approximate download size. Click **Start image search** and wait for the status line to report **Ready** — it also names the active compute backend (`webgpu` or `wasm`).

3. **Parity canary**: upload an image that is *in* the collection (any file from `objects/`). It must come back as the top result with a match score near 100%. This single check verifies that build-time and in-browser embeddings agree; if the top result is not the uploaded image (or a near-duplicate of it), the pipeline and page are out of sync.

4. Upload a non-collection image and confirm ranked results render with similarity badges, titles, and working links to item pages.

## evaluating models on your collection

The eval harness measures how reliably each model retrieves a collection image from a perturbed copy of itself (simulated crop, rotation, and photo degradation):

```sh
cd embeddings/scripts
node eval_retrieval.mjs                       # full matrix: 3 models × 2 profiles
node eval_retrieval.mjs --models dinov2,clip --profiles lineart --sample 40
```

Use it to pick the best `config-embeddings.yml` settings for your own material. Results for the original prototype collection are in the README.

## troubleshooting

- **`rake setup_embeddings` says Node is missing or too old** — install the LTS release from <https://nodejs.org/> (18 or newer), reopen your terminal, re-run.
- **Slow first `rake build_embeddings`** — the model weights (~12–90 MB depending on model) are downloaded once and cached; later runs are fast.
- **Page says "collection was built with … but this page expects …"** — the data in `assets/embeddings/data/` was generated by an older/different registry than the page code. Re-run `rake build_embeddings`.
- **Page status stuck on the model download** — the browser fetches model weights from the Hugging Face CDN on first visit (cached afterwards); check the network connection and the browser console.
- **Images skipped during build** — see `assets/embeddings/data/preprocess.log` for per-file reasons (usually a missing file at `object_location` or an unsupported extension).
- **Search results look uniformly similar / match percentages all high** — the preprocessing profile probably doesn't fit the material (e.g. `binary` on photographs). Switch to `standard` and rebuild.
