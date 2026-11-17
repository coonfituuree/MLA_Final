"""Fixed reused-benchmark comparison, never hyperparameter selection or tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_auc_score,accuracy_score,ConfusionMatrixDisplay
from shared.inference import Predictor,ROOT
from shared.text import prepare_text
from training.experiments import metrics
from training.prepare import save_json
from training.frozen_data import load_frozen_data

OUT=ROOT/'reports/improvements'

def evaluate(bundle=None):
    predictor=Predictor(bundle,require_complete=False)
    split=pd.read_csv(ROOT/'reports/splits.csv')
    assert (ROOT/'reports/splits.csv').read_bytes()==(OUT/'frozen_splits.csv').read_bytes()
    data,split=load_frozen_data()
    rows=data.iloc[split.loc[split.partition=='test','row_index']]
    texts=rows.apply(prepare_text,axis=1).tolist();y=rows.fraudulent.to_numpy(dtype=int)
    results={};errors=[]
    fig,ax=plt.subplots(figsize=(8,5))
    models=list(predictor.config['thresholds'])
    fig_cm,axes=plt.subplots(1,len(models),figsize=(4*len(models),4))
    for model,cm_ax in zip(models,axes):
        score=predictor.scores(predictor.matrix_for(texts,model),model)
        threshold=predictor.config['thresholds'][model]
        m=metrics(y,score,threshold)
        results[model]={'precision':m['precision'],'recall':m['recall'],'f1':m['f1'],'pr_auc_average_precision':m['average_precision'],'accuracy':float(accuracy_score(y,score>=threshold)),
                        'roc_auc':float(roc_auc_score(y,score)),'threshold':threshold,'confusion_matrix':m['confusion_matrix'],'false_positives':m['false_positives'],'test_count':len(y),'fraudulent_count':int(y.sum()),
                        'evaluation_status':'reused_original_benchmark_not_independent'}
        p,r,_=precision_recall_curve(y,score);ax.step(r,p,where='post',label=f'{model}: AP {m["average_precision"]:.3f}')
        ConfusionMatrixDisplay(np.array(m['confusion_matrix']),display_labels=['Legit','Fraud']).plot(ax=cm_ax,colorbar=False,cmap='Greens');cm_ax.set_title(model.replace('_',' '),fontsize=10)
        for kind,mask in [('false_positive',(y==0)&(score>=threshold)),('false_negative',(y==1)&(score<threshold))]:
            for i in np.flatnonzero(mask)[:4]:
                row=rows.iloc[i]
                errors.append({'model':model,'kind':kind,'job_id':int(row.job_id),'dataset_label':int(y[i]),'title':row.title,'score':float(score[i]),'source':'EMSCAD checksum in data/provenance.json; inspect original locally for text.'})
        print(model,results[model],flush=True)
    ax.set(xlabel='Recall',ylabel='Precision',title='Reused EMSCAD benchmark: descriptive comparison',ylim=(0,1.03));ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'precision_recall_v2.png',dpi=140);plt.close(fig)
    fig_cm.tight_layout();fig_cm.savefig(OUT/'confusion_v2.png',dpi=140);plt.close(fig_cm)
    save_json(predictor.bundle/'metrics.json',results)
    save_json(OUT/'benchmark_metrics.json',results);save_json(OUT/'benchmark_errors.json',errors)
    pd.DataFrame(results).T.drop(columns='confusion_matrix').to_csv(OUT/'benchmark_comparison.csv')
    if hasattr(predictor.logistic_features,'get_feature_names_out'):
        names=predictor.logistic_features.get_feature_names_out();coef=predictor.word_char_logistic.coef_[0];ix=np.r_[np.argsort(coef)[:20],np.argsort(coef)[-20:]]
        pd.DataFrame({'feature':names[ix],'coefficient':coef[ix]}).to_csv(OUT/'logistic_v2_coefficients.csv',index=False)
    return results

if __name__=='__main__':evaluate()
