# Executed verification record

Verified locally on Ubuntu using Python 3.12.16 in `.venv`, CPU PyTorch 2.7.1, Node 24.22.1, and npm 11.19.1. The official Kaggle download, both full training trials, saved artifacts, real HTTP requests and real browser requests were executed; the results below are not scaffolding or simulated predictions.

| Check | Result / evidence |
|---|---|
| Official dataset obtained | Official Kaggle API archive downloaded without credentials; CSV byte-identical to the supplied workspace file; SHA-256 `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`. Download script independently verifies it. |
| Data audit and full experiment | 17,880 raw rows, 15,872 usable unique advertisements; 4,795 connected groups; no identified group or 12,203 verified near-duplicate pairs crosses partitions. EDA, splits, quarantine and search records saved. |
| Models actually fitted | Majority DummyClassifier, three LR C values, two PyTorch MLP trials. Best neural epoch 21, validation AP .660319. Test AP .638921, F1 .610000. LR test AP .678647, F1 .625641. |
| Saved-artifact round trip | Training compares reloaded neural validation scores within 1e-7 and reloaded LR estimator outputs. Independent Predictor instances agree exactly on real held-out input; HTTP response agrees with direct Python inference. Manifest corruption is rejected. |
| Live health/readiness | Uvicorn with the saved bundle returns 200 for `/api/health`, `/api/ready`, and `/api/v1/model-info`; readiness reports CPU and emscad-v1. |
| Real POST requests | Structured real held-out text predicts through both models. Text-only real test job 358 produces neural score .9396876097 and threshold .7591350675; this differs from the full five-field evaluation input by design. |
| Invalid input | Missing, empty, whitespace, HTML-only/script-only descriptions, unknown model, extra target field and oversized input return 422; body-size limit returns 413, including streamed bodies. |
| File upload | Valid UTF-8 `.txt` follows the same inference pipeline as JSON description; wrong extension, empty text, invalid UTF-8, NULs and oversized upload fail with useful errors. |
| Missing artifacts | Test liveness stays 200, readiness and predictions return actionable 503. No random/mock fallback is present. |
| Python tests | **22 passed**, including parameterized cases, in `reports/tests.log`. One upstream Starlette/AnyIO deprecation warning; no test failures. |
| Frontend static checks | `npm --prefix frontend run typecheck` and `npm --prefix frontend run build` pass with the pinned dependency tree. |
| Actual browser integration | **1 Playwright scenario passed**, using installed Google Chrome, exercising the React UI → Vite `/api` proxy → FastAPI → loaded model. Both model scores match the Python fixture, backend validation is visible, upload succeeds, no page errors occur, and a 390px mobile viewport has no horizontal overflow. See `browser-results.json`. |
| Visual inspection | Desktop screenshot inspected after a real prediction. Desktop/mobile full-page screenshots saved in `reports/figures/`. |
| Native startup and cleanup | `scripts/start.sh` launched from `/tmp`; both local servers became available and a prediction through port 5173 matched the fixture. SIGTERM exits with 143 and reaps both owned server processes. No unrelated process is selected or killed. See `startup-verification.json` and `startup-test.log`. |
| Setup | `scripts/setup.sh` rerun successfully, using the pinned Python environment and `npm ci`. A fresh isolated Python environment then installed all 52 locked packages, loaded the bundle and reproduced the real score .9396876097. Setup explicitly resolves the two trusted public package indexes to avoid first-index shadowing; see `dependency-resolution.log` and `clean-environment.log`. |
| Evaluation command | `python -m training.evaluate` executed against fixed saved artifacts; regenerated metrics/figures agree with the recorded test results. No fitting/tuning occurs. |
| Course PDF and presentation | All PDF sections and rubric mapped in `COURSE_REQUIREMENTS.md`. Fifteen-slide PowerPoint generated and reopened successfully with python-pptx; visual slide rendering was not checked because LibreOffice is unavailable. Defense notes and AI disclosure included. |

## Limits of these checks

The browser fixture is one attributed real test example, used only for demonstration after final evaluation; it is not new independent evidence of generalization. Group detection is approximate. We did not evaluate score calibration, external current vacancies, non-English language, repeated seeds, or performance under concurrent load. The model misses 57 of 118 test frauds.

`npm audit` currently reports five high-severity development-tool dependency entries sharing the picomatch <=4.0.7 advisory, propagated through the installed latest Vite/plugin dependency tree. The registry's latest picomatch is 4.0.7 at this verification; automatically suggested downgrades would cross major versions. This remains a development-tool limitation; the app binds locally and no public deployment was performed. A clean audit is not claimed. Resolve/retest when a patched compatible upstream version is published.

Verification servers were stopped after the startup/cleanup test. Launch the application with `./scripts/start.sh`. Raw data and the isolated environments stay outside Git; trained artifacts, metrics, plots and reports are deliverables. Students must add real names/contributions and perform their own course defense.
