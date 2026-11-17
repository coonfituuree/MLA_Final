"""Describe fixed candidate differences without tuning any model or threshold."""
import json
import numpy as np
from shared.inference import Predictor,ROOT
from shared.text import prepare_text
from training.frozen_data import load_frozen_data
from training.prepare import save_json

def main():
    predictor=Predictor()
    if predictor.word_char_logistic is None:raise SystemExit('Select the complete v2 bundle before comparing errors.')
    data,split=load_frozen_data();rows=data.iloc[split.loc[split.partition=='test','row_index']]
    text=rows.apply(prepare_text,axis=1).tolist();truth=rows.fraudulent.to_numpy(dtype=int)
    outputs={}
    for model in ['logistic_regression','word_char_logistic']:
        scores=predictor.scores(predictor.matrix_for(text,model),model)
        outputs[model]={'scores':scores,'labels':(scores>=predictor.config['thresholds'][model]).astype(int)}
    old,new=outputs['logistic_regression'],outputs['word_char_logistic']
    changes=[]
    for i in np.flatnonzero(old['labels']!=new['labels']):
        row=rows.iloc[i]
        changes.append({'job_id':int(row.job_id),'title':row.title,'dataset_fraud_label':int(truth[i]),'old_label':int(old['labels'][i]),'new_label':int(new['labels'][i]),'old_score':float(old['scores'][i]),'new_score':float(new['scores'][i]),'effect':'corrected' if new['labels'][i]==truth[i] else 'introduced_error'})
    result={'scope':'already inspected original test benchmark; descriptive error analysis after validation selection; not independent generalization evidence',
            'changed_decisions':len(changes),'corrected':sum(r['effect']=='corrected' for r in changes),'introduced_errors':sum(r['effect']=='introduced_error' for r in changes),
            'changes':changes,'note':'Equal total recall can conceal different frauds detected. A small net false-positive gain does not establish calibration, semantic understanding, or illegal-recruitment detection.'}
    save_json(ROOT/'reports/improvements/error_changes.json',result)
    print({k:v for k,v in result.items() if k!='changes'})

if __name__=='__main__':main()
