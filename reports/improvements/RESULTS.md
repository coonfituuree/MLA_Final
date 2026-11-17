# Before/after results and diagnostic comparison

These results distinguish validation selection, the already inspected historical test benchmark, handwritten behavior probes and an uncertain user report. They are not an independent evaluation of illegal recruitment.

## Learned classifiers

Eight sparse trials selected word+character Logistic Regression (C=8, balanced weights). Validation AP: 0.659502 → 0.706492. Its threshold is 0.746093, chosen only from validation.

The English compact encoder was actually fine-tuned on 10,249 real training ads for up to 3 CPU epochs, with pretrained revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Selected epoch 1; validation AP 0.615438. It remains an explicit experimental option; the original MLP is preserved.

The following set is the **same 3,061-row original test benchmark**, already inspected during v1 development/error analysis. All new selection used validation; these numbers are descriptive reuse, not fresh independent-generalization evidence.

| Model | Precision | Recall | F1 | AP | False positives |
|---|---:|---:|---:|---:|---:|
| Original word LR | 0.7922 | 0.5169 | 0.6256 | 0.6786 | 16 |
| New word+character LR | 0.8026 | 0.5169 | 0.6289 | 0.6752 | 15 |
| Preserved MLP | 0.7439 | 0.5169 | 0.6100 | 0.6389 | 21 |
| Fine-tuned compact encoder | 0.8200 | 0.3475 | 0.4881 | 0.6351 | 9 |

AP is scikit-learn non-interpolated average precision. A slight F1 gain or a validation AP gain does not imply improved AP on every held-out distribution. No model or threshold was reselected using this benchmark, courier case, or equipment probes. Calibration was not evaluated.

## Diagnostic behavior

The complete original equipment advertisement/control were not supplied. The paired test uses the exact supplied paragraph in a clearly handwritten wrapper; only requirements differ. It is excluded from formal accuracy estimates.

| Diagnostic | Old neural score | Old LR score | New LR score | Fine-tuned encoder score | Heuristic | Overall review |
|---|---:|---:|---:|---:|---|---|
| equipment_paid | 0.072662 | 0.076574 | 0.051671 | 0.000946 | candidate_funded_equipment | additional_review_recommended |
| equipment_free | 0.411298 | 0.159348 | 0.129993 | 0.002406 | None | no_additional_signal_found |
| courier-user-2026-11-18 | 0.236349 | 0.830952 | 0.691638 | 0.002845 | None | additional_review_recommended |

Both original learned models and their thresholds are deliberately unchanged. The new character model is separately selectable and never silently substitutes for the original LR. Candidate-funded pre-agreement equipment is now separately highlighted for review; the employer-funded counterpart receives no heuristic warning. This is a transparent rule-based improvement in behavior, not learned fraud recall.

The exact translated courier advertisement was available and tested. Its original scores reproduce .236349 neural and .830952 LR. The user clarified that the alleged bait-and-switch comes from unsupplied online accounts, not independently documented evidence about this exact ad. It remains **uncertain**. Its heuristic layer does not infer illegal contents or flag the ordinary courier category; disagreement can prompt review without identifying criminal activity.

## Evidence and generalization limits

No suitable independently adjudicated illegal-recruitment dataset or untouched external evaluation set was obtained. Therefore **independent evaluation improvement is not established**. All training labels remain real EMSCAD fraud labels, with no synthetic training data or changed labels. See `data/evidence/README.md` for source types, provenance and privacy handling.

The frozen contextual review was audited on 2,562 real validation advertisements. Zero warnings were found, including the legitimate courier/remote/reimbursement/payment strata. This gives a narrow false-warning observation, but no positive risk recall. Nineteen diagnostic cases, including handwritten controls and the uncertain courier case, test software behavior rather than real-world accuracy.

See `SYSTEM_DIAGNOSIS.md`, `REVIEW_POLICY.md`, `original_sensitivity.json`, `heuristic_real_audit.json`, `benchmark_errors.json`, and the neural/PR/CM figures for supporting details.
