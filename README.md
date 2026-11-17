# VacancyLens

A working local university ML project for screening potentially fraudulent **English** job advertisements. React + TypeScript + Vite, FastAPI/Pydantic/Uvicorn, scikit-learn TF-IDF + Logistic Regression, and an actually trained PyTorch MLP. The neural network is the default; both trained models are selectable. Text and uploads remain local and are not persisted.

The current subtle-risk update is documented in the **v2 section below** and [update results](reports/improvements/RESULTS.md). The original v1 methodology remains archived for comparison.

## Quick start with included trained artifacts

Ubuntu, Node **22.12+ or 24 LTS**, npm, and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. Setup downloads an isolated Python **3.12**; it does not modify system Python. Tested with Python 3.12.16, Node 24.22.1, npm 11.19.1, and CPU PyTorch 2.7.1. Allow roughly 2 GB disk for environment/dependencies and a few GB RAM for training. No GPU or API key is required.

```bash
cd /home/coonfituuree/dev/MLA/final_project
./scripts/setup.sh
./scripts/start.sh
```

Open **http://localhost:5173**. Backend: **http://127.0.0.1:8000**, Swagger docs: **http://127.0.0.1:8000/docs**. Startup loads the saved bundle once and checks readiness; it never retrains. Ctrl+C stops only the processes this script starts. Both ports must be free. Source checkout location may change; scripts resolve the root themselves. Artifacts resolve relative to the project rather than the current terminal directory.

To install uv if absent, follow its linked installation instructions. An alternative with an already installed Python 3.12 is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
npm --prefix frontend ci
```

Setup tells uv to resolve exact versions across the two trusted public indexes (PyPI and the PyTorch CPU index), avoiding uv’s default first-index shadowing. `requirements.txt` declares direct dependencies; **requirements.lock.txt** pins the exact tested environment including transitive packages. Use the lockfile to load the supplied joblib artifacts consistently. Dependency metadata records exact inference library versions and fails clearly on mismatch. Only load the trusted project bundle: joblib is pickle-based. Manifest checks detect accidental corruption, not a maliciously replaced bundle and manifest.

## Obtain the actual dataset

```bash
.venv/bin/python scripts/download_data.py
```

This downloads `fake_job_postings.csv` from the [official Kaggle API archive](https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction), without credentials when available, and verifies SHA-256. Existing files are verified instead of downloaded again. On failure, manually download the CSV from [Shivam Bansal's Kaggle page](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place it at **data/raw/fake_job_postings.csv**, then rerun the command. Do not put raw data in Git. The original workspace copy in `backend/dataset/` was compared with the archive and copied to the canonical directory; future code uses only `data/raw/`.

Actual data: **EMSCAD, 17,880 × 18**, including **866 fraudulent and 17,014 legitimate ads**. CSV SHA-256 is `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`. Exact provenance, file size, retrieval date, license, and API metadata are in [data/provenance.json](data/provenance.json) and [data/kaggle_metadata.json](data/kaggle_metadata.json). Kaggle declares CC0; see the provenance/license discussion in the methodology. No synthetic training/evaluation data, SMOTE, or external LLM classifier is used.

## Reproduce training and evaluation

Run commands from the project root:

```bash
.venv/bin/python -m training.train
.venv/bin/python -m training.evaluate
```

The first command performs EDA, conflict quarantine, exact deduplication, employer/near-duplicate grouping, deterministic splitting, training-only TF-IDF fitting, a small validation search for LR and MLP, early stopping, validation-only threshold choice, artifact round-trip checks, and final held-out evaluation. It prints progress; the executed run is also recorded in `reports/training.log`. Full CPU training typically takes a few minutes on this machine; hardware will affect runtime. All **15,872 usable unique advertisements** participate in the experiment. The second command only re-evaluates fixed saved artifacts and thresholds; it neither trains nor tunes. Do not use repeated test evaluation to choose new hyperparameters.

Training overwrites `artifacts/emscad-v1/` and associated reports. Stop the backend before retraining and restart afterward. If changing the experiment, change `VERSION` in `training/train.py` to preserve the previous bundle. `artifacts/current.txt` selects the bundle. Optional backend configuration: `MODEL_BUNDLE=/absolute/path/to/trusted/bundle`; `MODEL_DEVICE=cpu` (default). The supplied bundle is CPU-compatible; CUDA is optional and requires a compatible separately installed PyTorch build and retraining/updated recorded environment.

The split is **10,249 train / 2,562 validation / 3,061 test**, with 474 / 121 / 118 frauds. Shared profiles/descriptions and verified five-word-shingle Jaccard near-duplicates stay within one partition. LSH is approximate and cannot detect every related ad. See [methodology/results](reports/METHODOLOGY.md) for exact methods, tuning configurations, explanations, and limitations.

## Actual held-out results

| Model | Fraud precision | Fraud recall | Fraud F1 | PR-AUC (average precision) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.0000 | 0.0000 | 0.0000 | 0.0385 | 0.5000 |
| Logistic Regression | 0.7922 | 0.5169 | 0.6256 | 0.6786 | 0.9267 |
| PyTorch MLP | 0.7439 | 0.5169 | 0.6100 | 0.6389 | 0.9256 |

Metrics are computed on the same test partition using frozen validation thresholds. PR-AUC means scikit-learn non-interpolated **average precision**, not trapezoidal area. The majority baseline has 96.1% accuracy while missing every fraud. Both learned models miss **57 of 118** frauds. LR outperforms the MLP here. Scores are **uncalibrated model scores**, not verified fraud probabilities; the result does not authenticate an employer. Historical 2012–2014 platform data cannot establish reliability on all current job advertisements.

## Run servers separately

Terminal 1, project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
npm --prefix frontend run dev
```

The Vite development proxy forwards relative `/api` requests to FastAPI. `npm --prefix frontend run build` type-checks and creates `frontend/dist`; it does not embed weights. `vite preview` is not the integrated development setup and does not provide the API proxy.

## API contract

- `GET /api/health`: process liveness.
- `GET /api/ready`: 200 only when the trusted bundle loads; actionable 503 otherwise.
- `GET /api/v1/model-info`: actual architecture, training configuration, thresholds, dataset provenance, and final metrics.
- `POST /api/v1/predictions`: JSON text fields.
- `POST /api/v1/predictions/file`: multipart `file` and optional `model`.

```bash
curl -fsS http://127.0.0.1:8000/api/ready
curl -fsS http://127.0.0.1:8000/api/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"title":"Software engineer","description":"Develop and maintain software with our engineering team. Experience with Python and automated testing required.","model":"neural_network"}'
curl -fsS http://127.0.0.1:8000/api/v1/predictions/file \
  -F 'file=@/absolute/path/to/vacancy.txt' -F 'model=logistic_regression'
```

The curl text is an illustrative user input, **not training or evaluation data**. Real held-out examples used for verification are recorded separately.

`description` is required and must remain readable after HTML removal. Optional `title`, `company_profile`, `requirements`, and `benefits` default to empty strings. `model` is `neural_network` (default) or `logistic_regression`. Unknown fields/models fail validation. Maximum title length 500; profile/requirements/benefits 20,000 each; total input 60,000 characters. Training and inference share the same combined-text preparation and 60,000-character cap. A `.txt` file is treated as description alone, up to 120,000 bytes, UTF-8 (BOM accepted), no NULs, also subject to the character limit. The HTTP body limit is 300 KB, including multipart overhead; large Unicode/escaped inputs may hit the byte limit before the character limit. Uploads are closed and discarded after the request. CPU inference runs in a thread pool; artifacts load once in the application lifespan.

Responses include `predicted_label`, human-readable `label`, `model_score`, `decision_threshold`, `model_name`, `model_version`, `score_note`, and `vocabulary_match` (whether any fitted vocabulary feature was found). Errors use FastAPI's `detail` field. Missing artifacts never produce mock or random predictions.

## Verification

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

For real browser integration, start both servers first, then:

```bash
.venv/bin/python scripts/browser_fixture.py
cd frontend
npx playwright install chromium
npm run test:e2e
```

On Ubuntu, browser system dependencies may be needed (`npx playwright install-deps chromium`). If Chrome is already installed, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/opt/google/chrome/chrome` instead of downloading Chromium. Browser configuration otherwise uses Playwright's installed Chromium. The browser test uses an attributed real held-out advertisement, checks both model scores against Python inference, exercises the Vite proxy, checks a backend validation error, uploads UTF-8 text, verifies mobile layout, and saves screenshots. It requires the dataset for fixture regeneration. The example fixture and evaluation cases are never used to tune the model after test evaluation.

See [reports/VERIFICATION.md](reports/VERIFICATION.md) for the actual verification record. Screenshots, metrics JSON/CSV, plots, and tests accompany the implementation.

## Project layout

```text
frontend/       React/TypeScript UI, Vite proxy, package-lock, browser test
backend/        FastAPI handlers and Pydantic schemas
shared/         shared text preparation, MLP architecture, artifact inference
training/       data audit/group split, training/search, final evaluation
scripts/        setup, dataset download, startup, fixture, slide generation
artifacts/      versioned trusted models/config/metrics/checksums
reports/        EDA, splits, tuning, curves, errors, methodology, slides, verification
data/           provenance/metadata; raw/ CSV excluded from Git
tests/          shared preprocessing, artifact integrity, API validation tests
```

## Course deliverables and attribution

Read [reports/METHODOLOGY.md](reports/METHODOLOGY.md), [reports/COURSE_REQUIREMENTS.md](reports/COURSE_REQUIREMENTS.md), [reports/DEFENSE_NOTES.md](reports/DEFENSE_NOTES.md), and the generated [presentation.pptx](reports/presentation.pptx). Rebuild the deck with `.venv/bin/python scripts/make_slides.py`. Review the slides and add the students' names and their actual contributions before submission. Every member must understand the pipeline and defend decisions; software cannot fulfill the oral-defense requirement on a student's behalf.

Dataset citation: Vidros S., Kolias C., Kambourakis G., Akoglu L. (2017), *Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset*, [doi:10.3390/fi9010006](https://doi.org/10.3390/fi9010006), via the linked Kaggle distribution. Dependencies and external documentation are cited in the methodology. The original v1 models used no pretrained weights or copied third-party model code; see the v2 encoder attribution below.

**AI assistance disclosure:** OpenAI Codex assisted with code, experiment execution, debugging, analysis, UI, documentation, and presentation materials. Metrics and examples are produced by the actual executable experiment. Students should retain this disclosure and review all content before presenting it.

## Subtle-risk review update (v2)

The updated API and UI separate the **learned EMSCAD fraud classification** from **additional heuristic review signals**. The selected model's original score/threshold/class fields remain available. Results also include both preserved primary model decisions, model-specific versions, exact evidence passages, processed-input digest and an overall review status. No scores are averaged. Technical comparison details are expandable.

Both original models and thresholds remain unchanged for compatibility. A separately selectable **word + character Logistic Regression** was actually trained in an eight-run validation-only search; it improves validation AP from .659502 to .706492. A compact English BERT experiment is genuinely fine-tuned with PyTorch on all real training advertisements, using all-text chunking. It is experimental and does not classify illegal recruitment. Actual results and the status of independent evaluation are in [reports/improvements/RESULTS.md](reports/improvements/RESULTS.md).

The review layer checks candidate-funded equipment combined with timing/seller/reimbursement context, personal-account receipt/forwarding of third-party money, and opaque delivery combined with evasive procedures. Warnings show exact supporting normalized text and are explicitly marked **HEURISTIC**. Ordinary courier duties, sealed parcels, remote work, high salaries, no experience and reimbursement alone are not triggers. “No additional signal found” never means verified or safe.

JSON and multipart requests accept optional `later_instructions` (up to 20,000 characters). It is reviewed separately and **excluded from learned advertisement features**. Evidence offsets refer to each HTML-stripped, whitespace-normalized field. Accepted advertisements are never silently truncated; a prepared text exceeding 60,000 characters is rejected, including field separators. The new `input_summary` reports length/digest and `truncated=false`.

The exact user-supplied courier text is recorded as **uncertain**, with the user's clarification that the possible document-to-illegal-delivery switch comes from online accounts whose sources were not provided. No criminal label is assigned. An initial advertisement may contain no clue to later recruiter instructions. See [evidence provenance](data/evidence/README.md), [diagnosis](reports/improvements/SYSTEM_DIAGNOSIS.md) and [review policy](reports/improvements/REVIEW_POLICY.md).

To reproduce v2 with the included original v1 bundle and frozen split:

```bash
./scripts/setup.sh
.venv/bin/python scripts/download_data.py
./scripts/reproduce_v2.sh
.venv/bin/python -m pytest -q
npm --prefix frontend run build
./scripts/start.sh
```

Fine-tuning needs network access only for its pinned public Google BERT initialization; saved serving artifacts run offline. CPU training may take tens of minutes on this laptop. The original `.venv/bin/python -m training.train` command reconstructs **v1** and selects its bundle; use `./scripts/reproduce_v2.sh` afterward to train/package v2. `python -m training.evaluate` dispatches to the active bundle's fixed evaluation without tuning. Stop the running app before training/package activation, then restart it; an already-running backend keeps its previously loaded artifacts.

For the browser diagnostic suite, regenerate the real fixture with `.venv/bin/python scripts/browser_fixture.py`, start both services, then run the documented Playwright command. The equipment fixture in `reports/improvements/browser-equipment.json` is clearly handwritten and never used as real accuracy evidence. Alternate QA ports are supported through `API_PROXY_TARGET` (Vite configuration) and `PLAYWRIGHT_BASE_URL`; normal startup remains localhost:5173 / 127.0.0.1:8000.

**Evaluation boundary:** the original 3,061-row test set was already inspected. It remains a fixed reused benchmark; no new untouched external risk dataset was obtained. Validation improvements and diagnostic passes do not establish independent improved generalization or illegal-drug recruitment detection. The heuristic audit has no independently adjudicated positive risk labels. See the update results for the actual false-warning observations and limitations.

**Updated attribution:** the optional encoder uses Google `bert_uncased_L-2_H-128_A-2`, Apache-2.0, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`, initialized from [its model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) and genuinely fine-tuned on EMSCAD; it is not an external inference API. The original v1 models used no pretrained weights. FTC/FBI guidance motivates some review checks, but warnings/news articles are not treated as labeled fraudulent advertisements. Codex assisted this update's implementation, experiments, diagnostics and documentation as disclosed above.

After startup, the model selector offers `neural_network`, `logistic_regression` (both original), `word_char_logistic` (new validation-selected sparse candidate), and `context_encoder` (fine-tuned experimental encoder). The active bundle's `available_models` is authoritative. The primary disagreement policy always refers to the preserved original models; a selected experimental model can also independently cross its own threshold. The app does not treat a larger score from another model as a more reliable probability.
