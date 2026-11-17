"""Generate concise before/after tables only from executed artifacts and diagnostics."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/improvements'

def main():
    old=json.loads((OUT/'original_metrics.json').read_text())
    new=json.loads((OUT/'benchmark_metrics.json').read_text())
    sparse=json.loads((OUT/'sparse_search.json').read_text())
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    encoder=json.loads((OUT/'encoder_search.json').read_text())
    lines=['# Before/after results and diagnostic comparison','',
    'These results distinguish validation selection, the already inspected historical test benchmark, handwritten behavior probes and an uncertain user report. They are not an independent evaluation of illegal recruitment.','',
    '## Learned classifiers','',
    f"Eight sparse trials selected word+character Logistic Regression (C=8, balanced weights). Validation AP: {sparse['original_validation']['average_precision']:.6f} → {sparse['selected']['validation']['average_precision']:.6f}. Its threshold is {sparse['selected']['validation']['threshold']:.6f}, chosen only from validation.",'',
    f"The English compact encoder was actually fine-tuned on {encoder['training']['train_count']:,} real training ads for up to {encoder['training']['epoch_limit']} CPU epochs, with pretrained revision `{encoder['source']['revision']}`. Selected epoch {encoder['selected']['best_epoch']}; validation AP {encoder['selected']['validation']['average_precision']:.6f}. It remains an explicit experimental option; the original MLP is preserved.",'',
    'The following set is the **same 3,061-row original test benchmark**, already inspected during v1 development/error analysis. All new selection used validation; these numbers are descriptive reuse, not fresh independent-generalization evidence.','',
    '| Model | Precision | Recall | F1 | AP | False positives |','|---|---:|---:|---:|---:|---:|']
    for label,m in [('Original word LR',old['logistic_regression']),('New word+character LR',new['word_char_logistic']),('Preserved MLP',new['neural_network']),('Fine-tuned compact encoder',new['context_encoder'])]:
        fp=m.get('false_positives',m['confusion_matrix'][0][1])
        lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['pr_auc_average_precision']:.4f} | {fp} |")
    lines+=['', 'AP is scikit-learn non-interpolated average precision. A slight F1 gain or a validation AP gain does not imply improved AP on every held-out distribution. No model or threshold was reselected using this benchmark, courier case, or equipment probes. Calibration was not evaluated.','',
    '## Diagnostic behavior','',
    'The complete original equipment advertisement/control were not supplied. The paired test uses the exact supplied paragraph in a clearly handwritten wrapper; only requirements differ. It is excluded from formal accuracy estimates.','',
    '| Diagnostic | Old neural score | Old LR score | New LR score | Fine-tuned encoder score | Heuristic | Overall review |','|---|---:|---:|---:|---:|---|---|']
    for case in diagnostics['cases']:
        if case['id'] not in ['equipment_paid','equipment_free','courier-user-2026-11-18']:continue
        codes=', '.join(s['code'] for s in case['review']['signals']) or 'None'
        lines.append(f"| {case['id']} | {case['before']['neural_network']['model_score']:.6f} | {case['before']['logistic_regression']['model_score']:.6f} | {case['after']['word_char_logistic']['model_score']:.6f} | {case['after']['context_encoder']['model_score']:.6f} | {codes} | {case['after']['neural_network']['overall_review']['status']} |")
    lines+=['', 'Both original learned models and their thresholds are deliberately unchanged. The new character model is separately selectable and never silently substitutes for the original LR. Candidate-funded pre-agreement equipment is now separately highlighted for review; the employer-funded counterpart receives no heuristic warning. This is a transparent rule-based improvement in behavior, not learned fraud recall.','',
    'The exact translated courier advertisement was available and tested. Its original scores reproduce .236349 neural and .830952 LR. The user clarified that the alleged bait-and-switch comes from unsupplied online accounts, not independently documented evidence about this exact ad. It remains **uncertain**. Its heuristic layer does not infer illegal contents or flag the ordinary courier category; disagreement can prompt review without identifying criminal activity.','',
    '## Evidence and generalization limits','',
    'No suitable independently adjudicated illegal-recruitment dataset or untouched external evaluation set was obtained. Therefore **independent evaluation improvement is not established**. All training labels remain real EMSCAD fraud labels, with no synthetic training data or changed labels. See `data/evidence/README.md` for source types, provenance and privacy handling.','',
    'The frozen contextual review was audited on 2,562 real validation advertisements. Zero warnings were found, including the legitimate courier/remote/reimbursement/payment strata. This gives a narrow false-warning observation, but no positive risk recall. Nineteen diagnostic cases, including handwritten controls and the uncertain courier case, test software behavior rather than real-world accuracy.','',
    'See `SYSTEM_DIAGNOSIS.md`, `REVIEW_POLICY.md`, `original_sensitivity.json`, `heuristic_real_audit.json`, `benchmark_errors.json`, and the neural/PR/CM figures for supporting details.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # A standalone learning figure for the genuinely fine-tuned encoder.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=encoder['selected']['history'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],marker='o');axes[0].set(xlabel='Epoch',ylabel='Weighted training BCE',title='Compact encoder: actual training')
    axes[1].plot([r['epoch'] for r in h],[r['average_precision'] for r in h],marker='o');axes[1].set(xlabel='Epoch',ylabel='Validation average precision',title='Validation checkpoint selection')
    fig.tight_layout();fig.savefig(OUT/'encoder_learning.png',dpi=150);plt.close(fig)
    print(OUT/'RESULTS.md')

if __name__=='__main__':main()
