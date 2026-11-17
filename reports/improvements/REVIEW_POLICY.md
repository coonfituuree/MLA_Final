# What the application recommends, and why

The top-level review status is a transparent decision policy, **not another learned classifier or combined probability**.

1. The advertisement's five trained fields are prepared once and supplied to both primary models. Each classifier uses its own features, saved weights and validation-selected threshold. The selected model's original response fields remain at the top level for compatibility.
2. Each submission is also passed through a separate rule-based review. Later recruiter instructions, when supplied, are reviewed separately and never change learned advertisement scores.
3. **Additional review recommended** means that the selected model or either primary model crosses its own threshold, or at least one contextual heuristic fires. Differences in primary class decisions are shown explicitly. Scores are never averaged or ranked as competing probabilities.
4. **No additional signal found** means only that the above policy found none. It does not verify the employer or assert safety. Every response sets `employer_verified=false` and `confirmed_illegal_activity=false`.

A warning links to a normalized field and exact start/end evidence offsets. In the UI it is labeled **HEURISTIC**, with explanation and, where applicable, rationale guidance links. Neither the evidence highlight nor this rationale is presented as neural-network reasoning.

## Supported contextual patterns

- Candidate-funded equipment **plus** pre-agreement timing, or candidate-funded equipment **plus** restricted seller **and** promised reimbursement. The presence of an equipment cost, reimbursement or company equipment policy alone does not suffice.
- Receiving third-party money **and** forwarding money **and** reference to a personal account in the same bounded context. Ordinary salary payments or company-controlled payment processing do not meet the full condition.
- Unclear/unchecked delivery contents **and** evasive procedures such as avoiding oversight or removing records. Courier work, sealed confidential documents, high salary, remote work and no-experience roles are not rules.
- Later instructions that explicitly describe a change and also meet one of the above contextual risk checks receive an additional compare-with-original warning. This is a limited targeted comparison, not general natural-language contradiction detection.

Action negation and coordinated negation are scoped to local text, with controls such as “never receive … or forward …” and “no payment is required”. A negated or employer-funded statement does not provide a positive candidate-payment assertion. Ordinary sealed parcels are not suspicious by themselves. Warning/advice document markers provide a separate document-kind hint and suppress instructional-heuristic interpretation; this hint is itself fallible, and adversarial or mixed warning/ad text may evade the checks.

## Boundaries

Heuristics operate within normalized fields and bounded sentence windows. They may miss references distributed across fields or long paragraphs, unusual wording, translations, ambiguous negation, or later changes not supplied to the system. A generic phrase can still be misunderstood. The original text may provide no clue to a later bait-and-switch. The review cannot infer contents, authenticate recruiter messages, identify drugs, establish illegality, or verify independent accounts.

The policy flags model disagreement for review, never as proof that the higher-scoring model is right. The new word+character LR remains separately selectable, with its validation-selected threshold; the original primary models and their decisions are retained for compatibility. The experimental context encoder is shown only when available and runs only when selected; it does not silently replace the primary disagreement policy. Its learned score concerns the same EMSCAD fraud labels, not criminal recruitment.

## Evidence and evaluation

See `data/evidence/README.md`, `heuristic_real_audit.json`, `heuristic-audit-lock.json` and `SYSTEM_DIAGNOSIS.md`. Rules are tested with clearly handwritten behavior probes and separately audited on real source-identified advertisements. There are no independently adjudicated labels for the targeted criminal-recruitment patterns. We therefore report coverage and observed false-warning behavior, **not independently validated risk accuracy**.
