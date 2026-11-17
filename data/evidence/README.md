# Evidence and diagnostic data policy

These records preserve the difference between advertisements, user reports, and educational warnings. They are not a new training dataset.

- `courier_user_case.json`: exact English text supplied by the user, ending in DeepL attribution. The original-language advertisement, source URL/date and recruiter messages are unavailable. The user clarified that online accounts describe document-delivery ads followed by a different illegal-delivery offer, and that those accounts are not independent evidence about this exact advertisement. **Label: uncertain.** Used only for diagnosis and model sensitivity; never training, selection, threshold tuning or accuracy claims. No criminal labels are inferred from scores, salary, geography, courier duties or sealed parcels.
- `emscad_controls.json`: short attributable quotations from real EMSCAD training advertisements, with source job IDs, excerpt hashes and original historical fraud labels. These are qualitative negative controls, not newly verified employers. Full advertisements remain in ignored `data/raw/`; report excerpts exclude personal contact details.
- `diagnostics/cases.py`: explicitly handwritten behavioral probes. One reproduces the user's exact equipment paragraph inside a clearly handwritten professional wrapper, paired with employer-funded equipment with every other field held constant. These are **not real labeled advertisements**, are excluded from all training and formal model evaluation, and do not reproduce the unavailable complete Junior Operations Coordinator advertisements. Expected warnings are software behavior assertions, not fraud ground truth.

The actual supervised training and formal classifier comparisons use only the real EMSCAD CSV, unchanged fraud labels and frozen group split. Provenance and license remain in `data/provenance.json`. There is no validated illegal-recruitment training/evaluation set. The already inspected original test set is only a reused descriptive benchmark; no new independent generalization result is claimed.

## Rationale sources, not advertisement examples

- [FTC, Looking for a job? Scammers might be looking for you (January 2023)](https://consumer.ftc.gov/consumer-alerts/2023/01/looking-job-scammers-might-be-looking-you): describes job offers requiring payment for equipment with promised payroll reimbursement. Used to motivate contextual review of candidate funding plus timing/seller constraints, not to label every reimbursement arrangement fraudulent.
- [FBI, Money Mules](https://www.fbi.gov/how-we-can-help-you/common-frauds-and-scams/money-mules): discusses recruitment involving personal accounts and movement of third-party funds. Used to motivate a combined personal-account/receive/forward check. Ordinary company-account payment processing and payroll do not meet this condition.

These government pages are **warnings/guidance**, not fraudulent advertisements. They are never supplied as positive job advertisements to a classifier or training pipeline. Delivery opacity plus evasive procedures is an explicitly conservative engineering heuristic, with no claimed criminal ground truth or measured drug-detection capability. Ordinary sealed/confidential documents are a negative control.

## Scope and privacy

Evidence passages returned by the API are exact substrings of the indicated HTML-stripped, whitespace-normalized input field, with start/end offsets. The original text is never reconstructed or completed by a language model. Reports redact contact patterns before storing excerpts; offsets in a redacted report refer to the original local normalized field. API text remains ephemeral and is never added to training automatically.

A separate `later_instructions` field preserves provenance within a submission. It is excluded from all learned-model features, because EMSCAD training inputs are advertisements rather than authenticated conversations. Its warnings cannot establish that a message is genuine or a role is illegal. A bait-and-switch occurring after contact can be entirely invisible in the original advertisement.
