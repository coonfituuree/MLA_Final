# Executed update verification

- Real EMSCAD source checksum and original split preserved; original v1 artifact checksums unchanged.
- Eight real-label sparse experiments completed. Tiny English BERT fine-tuned for three full-data epochs; validation restores epoch 1. Embedding and attention-weight changes verified against the pinned pretrained initialization (`encoder_weight_change.json`).
- Active bundle: emscad-v2, with four selectable fitted classifiers. All 17 manifest entries verified, including offline tokenizer and trained state_dict.
- **51 Python checks passed**, with no skipped optional-model checks. Covers complete input to both primary models, boundary rejection, preserved predictions after reload, API/file validation, evidence offsets, negation, company-account/payroll distinction, all-token encoder chunking, and later instructions excluded from learned scores.
- **Four real-browser scenarios passed** through React → Vite proxy → FastAPI → saved artifacts, including all four classifiers, latest full form values, controlled equipment pair, exact uncertain courier case, validation errors, UTF-8 upload and mobile layout.
- TypeScript checking and production build passed. HTTP health/readiness and model-info passed on the final v2 backend.
- Six source-identified real legitimate counterexamples and 19 labeled diagnostic behavior cases recorded. Diagnostic passes are not real-world accuracy estimates. The frozen real-ad audit has zero warnings on 2,562 validation advertisements and cannot estimate positive risk recall.
- Seven-slide update presentation generated and reopened; visual slide rendering not tested. Desktop/mobile screenshots saved.

QA used localhost:5174 and 127.0.0.1:8001 because the user's existing v1 process occupied the normal ports. Only owned QA processes were stopped afterward; the existing user process was left untouched. **Stop the earlier app with Ctrl+C and run `./scripts/start.sh` to load v2 on localhost:5173 / 127.0.0.1:8000.** A running FastAPI process deliberately does not reload artifacts automatically.

No suitable independent illegal-recruitment labels or untouched external evaluation set were obtained. Original test results are explicitly a reused descriptive benchmark, not new independent generalization evidence. Contextual warnings are heuristic and do not authenticate employers or establish crimes. The original project's development-dependency audit limitations remain documented in `reports/VERIFICATION.md`.
