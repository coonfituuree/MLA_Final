"""Validation-only sparse-model experiments; original split and v1 artifacts remain immutable."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix
from shared.text import prepare_text
from training.prepare import ROOT, save_json
from training.frozen_data import load_frozen_data
from training.train import threshold_for_f1

OUT=ROOT/'artifacts/experiments/sparse-v2'

def metrics(y,scores,threshold):
    p,r,f,_=precision_recall_fscore_support(y,scores>=threshold,average='binary',zero_division=0)
    return dict(precision=float(p),recall=float(r),f1=float(f),average_precision=float(average_precision_score(y,scores)),false_positives=int(((scores>=threshold)&(y==0)).sum()),confusion_matrix=confusion_matrix(y,scores>=threshold).tolist(),threshold=float(threshold),count=len(y),positive_count=int(y.sum()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split_path=ROOT/'reports/splits.csv'
    split=pd.read_csv(split_path)
    assert split_path.read_bytes()==(ROOT/'reports/improvements/frozen_splits.csv').read_bytes()
    data,split=load_frozen_data()
    data['text']=data.apply(prepare_text,axis=1)
    rows={p:data.iloc[split.loc[split.partition==p,'row_index']] for p in ['train','validation']}
    y={p:rows[p].fraudulent.to_numpy(dtype=int) for p in rows}
    trials=[];best=None
    for family in ['word','word_char']:
        features=[('word',TfidfVectorizer(max_features=12000,min_df=2,max_df=.98,ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode',dtype=np.float32))]
        if family=='word_char':
            features.append(('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=18000,min_df=3,sublinear_tf=True,dtype=np.float32)))
        vectorizer=FeatureUnion(features)
        train=vectorizer.fit_transform(rows['train'].text)
        val=vectorizer.transform(rows['validation'].text)
        for c,weight in [(2.,'balanced'),(8.,'balanced'),(2.,None),(8.,None)]:
            model=LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=700,random_state=42)
            model.fit(train,y['train']);score=model.predict_proba(val)[:,1]
            threshold=threshold_for_f1(y['validation'],score)
            result=dict(family=family,C=c,class_weight=weight,validation=metrics(y['validation'],score,threshold))
            trials.append(result);print(json.dumps(result),flush=True)
            if best is None or result['validation']['average_precision']>best['validation']['average_precision']:
                best=result;joblib.dump(vectorizer,OUT/'features.joblib');joblib.dump(model,OUT/'estimator.joblib')
    old_vector=joblib.load(ROOT/'artifacts/emscad-v1/vectorizer.joblib')
    old_lr=joblib.load(ROOT/'artifacts/emscad-v1/logistic_regression.joblib')
    old_config=json.loads((ROOT/'artifacts/emscad-v1/config.json').read_text())
    original=metrics(y['validation'],old_lr.predict_proba(old_vector.transform(rows['validation'].text))[:,1],old_config['thresholds']['logistic_regression'])
    save_json(OUT/'selection.json',dict(selected=best,original_validation=original,trials=trials,seed=42,split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),test_used_for_selection=False,warning='Original test outcomes were previously inspected. Reused benchmark results are descriptive, not independent new generalization evidence.'))
    save_json(ROOT/'reports/improvements/sparse_search.json',json.loads((OUT/'selection.json').read_text()))

if __name__=='__main__':main()
