# System diagnosis and evidence boundaries

## Confirmed trace

- `VacancyInput` receives all five trained fields; `prepare_text` combines title, company profile, description, requirements and benefits in that order. The frontend submits its current React state. No stale-form defect was found. Browser assertions now compare the actual outgoing JSON fields against the form and check a backend digest of the prepared input.
- The two original classifiers share exactly the same prepared text and fitted word vectorizer. A separately selectable Logistic Regression adds character features on that same complete text; it does not receive extra information. Both original primary models retain their original vectorizer, weights and thresholds for compatibility; the new candidate does not silently replace the original LR. The optional compact encoder chunks the same full text rather than keeping only the first tokens.
- Class 1 is fraudulent. The MLP emits one logit, with one sigmoid applied in `neural_scores`. `eval()` and `torch.inference_mode()` are used. LR uses class index 1 in `predict_proba`. Trusted artifact checksums, preprocessing and library versions are checked during lifespan loading. No mapping, double-sigmoid or reload-per-request defect was found.
- Old thresholds are neural .759135 and LR .697014, selected by validation fraud F1. Positive neural loss weight is 9775/474 = 20.6224; LR uses balanced class weights. Therefore the scores are not calibrated probabilities and numerical magnitudes cannot be compared across models.
- The existing group split is preserved byte for byte: SHA-256 `4abc3a4449456421743d3f7a2dac6e69ff54e1af31ddefa5c1c8afce7f2a2b7a`. Training-only features and unchanged real labels remain mandatory. The original approximate duplicate/employer grouping prevents detected group overlap but is not a proof against all leakage.

## Confirmed implementation bug and fix

The old endpoint limited the sum of raw field lengths to 60,000, while `prepare_text` silently sliced the concatenation at 60,000. Field separators could push a boundary-sized submission beyond that limit and drop its final characters. It did **not** explain the supplied courier case or normal-sized equipment paragraph: the longest prepared EMSCAD advertisement is only 14,845 characters.

The validator now checks the complete normalized concatenation, including separators, and rejects an over-limit submission clearly. Direct inference enforces the same boundary. Legacy text-v1 preparation remains compatible, and accepted HTTP inputs never reach its clipping branch. The response reports prepared character count, SHA-256, included fields and `truncated=false`. A long-text regression checks a final benefits sentinel reaches both primary feature pipelines.

## Established model limitation versus hypotheses

The original neural trial's training loss falls to .0032 while validation weighted loss rises to 1.6495. Validation AP peaks at .6603; its original test AP is .6389. The original LR test AP is .6786. These are observed signs of overfitting and limited historical generalization, rather than evidence of an inference bug. TF-IDF models do not explicitly encode the relationships among who pays, contract timing, seller restriction and reimbursement. Sparse word associations and limited positive examples are plausible reasons for the counterintuitive paired behavior, but no experiment proves a unique cause.

The new source-controlled experiment tests word versus word+character n-grams, C = 2/8 and balanced versus unweighted loss. All eight combinations use validation only. Selected word+character LR validation AP is .706492 versus .659502 originally. This is a measured validation improvement, not independent external accuracy. The compact encoder experiment genuinely fine-tunes a pretrained English text model with PyTorch and real EMSCAD labels; its actual results, including failures, are recorded separately.

## Exact courier case

The user-supplied translated advertisement exactly reproduces the old scores: neural **.2363489866**, LR **.8309520002**. With their own thresholds, these classify differently. We have the exact English text, but not the original-language ad, source/date or later messages.

The user clarified that online accounts describe recruiters advertising document delivery and later offering illegal drug delivery after claiming the original roles are filled. No sources supporting that pattern or independent evidence about this exact ad have been supplied. The evidence record labels it **uncertain**, not a confirmed illegal-recruitment example. A genuine bait-and-switch may leave no identifying sign in the original ad.

Old LR's largest positive local contributions include “money”, “earn”, “jobs”, “flexible schedule” and “part time”. These explain LR's additive logit for this input, **not criminal intent**. Ordinary legitimate ads can contain all these terms. The MLP sensitivity test finds that deleting the long keyword-stuffed closing paragraph raises its score by about .237, while deleting other passages can shift it again. This demonstrates representation sensitivity; deletion also changes TF-IDF normalization and is not a causal or semantic explanation. The full reproducible LR contributions and MLP sentence-deletion deltas are in `original_sensitivity.json`.

The review layer deliberately does **not** flag the courier case solely for sealed parcels, income, quick onboarding, brand keywords or no experience. The overall review status recommends checking it because the preserved primary classifiers disagree at their own thresholds. The new word+character candidate itself scores this case about .692, below its .746 threshold; this is not used to choose or alter any model, threshold or label. That is not a claim that the higher score is right, and it identifies no illicit substance.

## Equipment case

Only the suspicious equipment paragraph was supplied, not the complete professional advertisement or exact free-equipment control. The old .001/.036 full-ad scores cannot be reproduced faithfully. We preserve the supplied paragraph verbatim and put it in a clearly handwritten controlled wrapper; the free-equipment version changes **only requirements**, with all other fields constant. It is a diagnostic test, not training data or real-world accuracy evidence.

In that wrapper the old MLP scores the candidate-paid version .07266 and the employer-paid control .41130. This counterintuitive ranking is reported rather than “fixed” by changing labels or selecting a threshold on it. The separate contextual review warns on candidate payment combined with timing/seller/reimbursement relationships and does not warn on the free-equipment control. The learned score remains exactly what the model computes.

## Review layer and evaluation limits

The review layer is explicitly **heuristic**, with exact normalized-field evidence spans. It is not neural reasoning and is not trained on criminal recruitment labels. It requires contextual combinations, scopes negation, avoids ordinary job-category triggers and treats warning/advice documents separately. Later instructions are supplied separately and excluded from the learned fraud features.

Rule v1 was frozen before an audit; v1.1 fixes tail coverage in long unpunctuated fields; v1.2 requires receipt and personal-account use in the same sentence so separate payroll information cannot contaminate company-payment duties. Earlier audits are preserved and subsequent audits explicitly reuse the same partition. Rules are frozen before each audit of all 2,562 real validation ads, with their source hash stored. The audit produces **zero warnings**, including zero on five dataset-legitimate courier ads, 94 legitimate remote ads, 68 legitimate reimbursement ads, and 25 legitimate payment-processing ads. This supports the narrow observation that these checks did not blanket-flag those ads. It provides **no positive-case risk recall**, and no independently adjudicated labels exist for the targeted risk patterns. Handwritten tests measure software behavior only. Do not quote their pass count as detection accuracy.

The original test set has already been inspected during the original project and error analysis. It is kept as a fixed **reused descriptive benchmark**. All new model choices use validation, but no new independent-generalization claim is made. We did not obtain a suitable independently verified illegal-recruitment dataset or a fresh adjudicated external holdout. Obtaining one is necessary before asserting improved detection of that task.

Additional source-identified legitimate controls include job 4914, “A new laptop to work on is provided to each new employee,” and job 964, an accounts-payable role that prepares and mails checks. Both are real historical training advertisements labeled 0 in EMSCAD and receive no heuristic warning. Those dataset labels do not independently verify present-day employers; these are qualitative counterexamples, not a new risk accuracy set.
