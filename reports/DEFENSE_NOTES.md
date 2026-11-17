# Presentation and defense notes

Use `presentation.pptx` with the working app and `METHODOLOGY.md`. Add the actual student names and contribution breakdown before submission. The course allows up to three members. Each member should explain every major step; the PDF prohibits holding phones during the defense.

## Suggested 8–10 minute narrative

1. **Problem and motivation:** recruitment fraud threatens money and personal information. We build text screening, with 0 legitimate and 1 fraudulent. The model does not authenticate an employer.
2. **Real data:** official EMSCAD via Kaggle, byte-verified checksum, 17,880 historical English ads, only 866 frauds. Explain geography, platform, age, and expert-label limitations.
3. **EDA changes our choices:** missing optional fields are common, fraud text is shorter on average, and accuracy hides failures. Show the EDA figure and explain overlap rather than claiming a simple rule.
4. **Leakage prevention:** remove same-label exact duplicate inputs; quarantine conflicts; connect long shared company profiles, descriptions, and verified near-duplicates. Keep groups intact and fit vocabulary/IDF only on training. Group balance is approximate.
5. **Models:** the majority class sets an honest floor; LR supplies a strong sparse-text baseline and global coefficients; MLP demonstrates learned nonlinear feature combinations with genuine backpropagation.
6. **Neural learning:** sparse minibatches become dense; logits go to weighted BCEWithLogitsLoss; gradients update weights through ReLU with AdamW. Dropout and weight decay regularize; validation AP chooses the checkpoint.
7. **Tuning and thresholds:** only validation chooses the fixed small grid, early stopping, and F1 threshold. Test labels are opened after decisions. No test-driven retuning.
8. **Results and surprises:** LR F1 .626 versus neural .610; both find 61/118 frauds. Majority accuracy 96.1% is useless for fraud recall. Explain AP versus operating-point metrics.
9. **Failure analysis:** show false positives and negatives. Training/validation loss separation demonstrates overfitting; strong company/industry coefficients indicate possible shortcuts. LR coefficients do not explain the neural network.
10. **Demo and limitations:** show both selectable models, a file upload, and a validation error. End with human review and a need for fresh independently labeled data, calibration, and cost-based thresholds.

## Questions to rehearse

**What is TF-IDF?** Term frequency gives weight to terms in an advertisement; inverse document frequency reduces weights of terms common across training documents. Sublinear TF uses 1 + log(count). Word unigrams and bigrams are selected into a 12,000-term vocabulary; rows are L2-normalized. Validation/test never alter vocabulary or IDF.

**Why no stemming or aggressive cleanup?** Job titles, names, punctuation, and wording carry information. HTML and whitespace normalization remove irrelevant formatting without rewriting the language. The vectorizer supplies its standard word tokenization.

**Why class weighting?** About 95% of original examples are legitimate. Neural positives receive a training-only weight of 20.6224, and LR uses balanced class weights. No examples are synthetically generated. Weighted scores need not be calibrated probabilities.

**How does the MLP learn?** Hidden activations are ReLU(Wx+b), dropout randomly suppresses activations during training, and the output is one logit. Weighted binary cross-entropy provides a differentiable error; backpropagation computes gradients; AdamW updates parameters. Sigmoid converts logits into scores at inference. Explain the actual code in `shared/model.py` and `training/train.py`.

**What exactly is early stopping?** Monitor validation average precision after each epoch. Save weights when improvement exceeds .0001. Stop after five epochs without such improvement. Restore the selected weights from epoch 21, not the final epoch 26. This criterion optimizes ranking, not validation loss.

**What is AP/PR-AUC?** Average precision weights precision by increments in recall across score thresholds. It is scikit-learn average_precision_score, not trapezoidal PR area. ROC-AUC is ranking of positives above negatives; abundant negatives can make ROC look strong while recall/precision at a practical threshold remain weak.

**Why is the threshold not .5?** It maximizes fraudulent-class F1 on validation only. Neural .759135 and LR .697014 reflect their score scales and validation tradeoffs. Business costs are unknown; F1 is a documented academic operating criterion, not an optimal production policy.

**How do you prevent leakage?** Explain exact deduplication, conflict quarantine, connected groups, group-aware split, train-only TF-IDF and weights, validation-only choices, final test. LSH is approximate and employer profiles are imperfect proxies: never claim leakage is impossible.

**Why did LR win?** The MLP has about 1.54 million parameters with only 474 unique training frauds, while LR has fewer parameters and a strong sparse-text bias. Validation loss rises as training loss falls. This is evidence consistent with overfitting, not proof of every cause. The chosen small search did not explore all alternatives.

**Can we trust a low score?** No. The neural model misses 57 of 118 held-out frauds. A score is an uncalibrated output of a historical text classifier, and legitimate-looking language can be fraudulent. Verify employers separately.

**What would you do next?** Collect fresh independent labels, validate across employers/time/geography, measure calibration on new validation data, use stakeholder costs for thresholds, audit shortcut features, quantify uncertainty with grouped resampling. Keep a new untouched test set for future decisions.

## Demo commands

Start with `./scripts/start.sh`. Open localhost:5173 and show the actual model-information panel. Use a real held-out example from `reports/browser-fixture.json` only as an already evaluated demonstration, never for tuning. To reproduce this fixture, run `.venv/bin/python scripts/browser_fixture.py`. Select each model and observe scores, upload a UTF-8 text file, and show an empty/HTML-only input error. Explain that text-only uploads omit optional fields, so scores can differ from a full structured advertisement.

## Code walkthrough

- `shared/text.py`: one preprocessing contract used everywhere.
- `training/prepare.py`: deduplication, conflict handling, group detection, saved split.
- `training/train.py`: fitting, search, minibatch backpropagation, early stopping and threshold selection.
- `training/evaluate.py`: fixed test evaluation and reports, no tuning.
- `shared/inference.py`: trusted bundle, eval/inference mode, consistent transformations.
- `backend/main.py` and `schemas.py`: lifespan loading, validation, thread-pool inference, file handling.
- `frontend/src/App.tsx`: relative API calls, model selection, real evaluation, loading and errors.

## Attribution and contributions

Cite EMSCAD (Vidros et al. 2017), its Kaggle distribution, and the library documentation listed in the methodology. Explicitly disclose that OpenAI Codex assisted with implementation, running experiments, debugging, analysis, design, and written/slide materials. State each student's real contribution and the parts they independently reviewed or modified. Do not claim AI-generated work was independently authored without assistance.
