"""Evaluate fixed artifacts and thresholds; never tune against test labels."""
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
from shared.text import prepare_text
from shared.inference import Predictor
from training.prepare import ROOT, save_json

def evaluate(require_complete=True):
    predictor = Predictor(require_complete=require_complete)
    if predictor.config.get('training',{}).get('word_char_v2'):
        from training.evaluate_v2 import evaluate as evaluate_current
        return evaluate_current(predictor.bundle)
    df = pd.read_csv(ROOT/'data/raw/fake_job_postings.csv').fillna('')
    split = pd.read_csv(ROOT/'reports/splits.csv')
    heldout = split.loc[split.partition == 'test']
    rows = df.iloc[heldout.row_index].copy()
    y = rows.fraudulent.to_numpy(dtype=int)
    matrix = predictor.vectorizer.transform(rows.apply(prepare_text, axis=1))
    metrics, errors = {}, []
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    pr_fig, pr_ax = plt.subplots(figsize=(7,5))
    for ax, name in zip(axes, ['majority_baseline','logistic_regression','neural_network']):
        scores = predictor.scores(matrix, name)
        threshold = predictor.config['thresholds'][name]
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
        cm = confusion_matrix(y, predictions, labels=[0,1])
        metrics[name] = dict(accuracy=float(accuracy_score(y,predictions)),precision=float(precision), recall=float(recall), f1=float(f1), pr_auc_average_precision=float(average_precision_score(y,scores)),roc_auc=float(roc_auc_score(y,scores)),threshold=threshold,confusion_matrix=cm.tolist(),test_count=len(y),fraudulent_count=int(y.sum()))
        ConfusionMatrixDisplay(cm, display_labels=['Legit','Fraud']).plot(ax=ax, colorbar=False, cmap='Greens')
        ax.set_title(name.replace('_',' ').title(), fontsize=10)
        p,r,_ = precision_recall_curve(y,scores)
        pr_ax.step(r,p,where='post',label=f'{name} (AP {metrics[name]["pr_auc_average_precision"]:.3f})')
        for kind, mask in [('false_positive',(y==0)&(predictions==1)),('false_negative',(y==1)&(predictions==0)),('true_positive',(y==1)&(predictions==1)),('true_negative',(y==0)&(predictions==0))]:
            # Fixed row order, not cherry-picked for an impressive story.
            for i in np.flatnonzero(mask)[:3]:
                row = rows.iloc[i]
                errors.append(dict(model=name,kind=kind,job_id=int(row.job_id),true_label=int(y[i]),score=float(scores[i]),title=row.title,excerpt=prepare_text(row.to_dict())[:1800]))
    fig.tight_layout(); fig.savefig(ROOT/'reports/figures/confusion_matrices.png',dpi=150); plt.close(fig)
    pr_ax.axhline(y.mean(),ls=':',color='gray',label='Test prevalence')
    pr_ax.set(xlabel='Fraudulent-class recall',ylabel='Fraudulent-class precision',title='Held-out precision–recall curves',ylim=(0,1.03))
    pr_ax.legend(fontsize=8); pr_fig.tight_layout(); pr_fig.savefig(ROOT/'reports/figures/precision_recall.png',dpi=150); plt.close(pr_fig)
    history = pd.read_json(ROOT/'reports/neural_history.json')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(history.epoch,history.train_loss,label='Training (dropout active)')
    axes[0].plot(history.epoch,history.validation_loss,label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Weighted BCE',title='Selected neural trial: loss'); axes[0].legend()
    axes[1].plot(history.epoch,history.validation_ap,color='#2f776c')
    axes[1].set(xlabel='Epoch',ylabel='Average precision',title='Validation metric for early stopping')
    fig.tight_layout(); fig.savefig(ROOT/'reports/figures/neural_learning.png',dpi=150); plt.close(fig)
    names = predictor.vectorizer.get_feature_names_out()
    coefs = predictor.logistic.coef_[0]
    top = np.r_[np.argsort(coefs)[:15],np.argsort(coefs)[-15:]]
    pd.DataFrame({'term':names[top],'coefficient':coefs[top]}).to_csv(ROOT/'reports/logistic_coefficients.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,8)); ax.barh(names[top],coefs[top],color=['#2f776c' if c<0 else '#bd7043' for c in coefs[top]])
    ax.set(title='Logistic Regression only: strongest coefficients',xlabel='Coefficient (not a causal explanation)'); fig.tight_layout(); fig.savefig(ROOT/'reports/figures/logistic_coefficients.png',dpi=150); plt.close(fig)
    save_json(ROOT/'reports/metrics.json',metrics)
    pd.DataFrame(metrics).T.drop(columns='confusion_matrix').to_csv(ROOT/'reports/comparison.csv')
    save_json(ROOT/'reports/error_examples.json',errors)
    save_json(predictor.bundle/'metrics.json',metrics)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in predictor.bundle.iterdir() if p.is_file() and p.name!='manifest.json'}
    save_json(predictor.bundle/'manifest.json',manifest)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    evaluate()
