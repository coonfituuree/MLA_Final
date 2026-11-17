"""Repeatable before/after diagnostics and ablations, not an accuracy benchmark."""
import hashlib
import json
import re
from copy import deepcopy
import numpy as np
import pandas as pd
from shared.inference import Predictor, ROOT
from shared.text import FIELDS, clean_text, prepare_text
from shared.review import review_record
from diagnostics.cases import CASES
from training.prepare import save_json

OUT=ROOT/'reports/improvements'

def ablation(predictor,record):
    text=prepare_text(record)
    matrix=predictor.vectorizer.transform([text])
    names=predictor.vectorizer.get_feature_names_out()
    contributions=matrix.toarray()[0]*predictor.logistic.coef_[0]
    active=matrix.indices
    ranked=sorted(active,key=lambda i:abs(contributions[i]),reverse=True)[:16]
    base=predictor.predict(record,'neural_network')['model_score']
    sensitivity=[]
    for field in FIELDS:
        value=clean_text(record.get(field))
        if not value:continue
        pieces=[m for m in re.finditer(r'[^.!?]+(?:[.!?]+|$)',value) if m.group().strip()]
        # At most 30 sentence deletions, selected in source order, plus whole-field deletion.
        for match in pieces[:30]:
            modified=deepcopy(record);modified[field]=value[:match.start()]+' '+value[match.end():]
            if not prepare_text(modified):continue
            score=predictor.predict(modified,'neural_network')['model_score']
            sensitivity.append({'field':field,'removed_normalized_text':match.group().strip(),'original_score':base,'score_without_passage':score,'delta_original_minus_ablated':base-score})
    return {'model_version':predictor.config['model_version'],'logistic_features':[{'term':names[i],'tfidf_times_coefficient':float(contributions[i])} for i in ranked],
            'logistic_intercept':float(predictor.logistic.intercept_[0]),'neural_sentence_ablation':sorted(sensitivity,key=lambda x:abs(x['delta_original_minus_ablated']),reverse=True)[:10],
            'limitations':'LR contributions are LR-only additive logit contributions. Neural deletion recomputes TF-IDF and normalization; deltas reflect sensitivity to deletion, not causal reasoning, semantic understanding, or criminal evidence.'}

def main():
    old=Predictor(ROOT/'artifacts/emscad-v1');current=Predictor()
    courier=json.loads((ROOT/'data/evidence/courier_user_case.json').read_text())
    cases=CASES+[dict(id=courier['id'],fields=courier['fields'],source_type=courier['source_type'],label_status=courier['evidence_status'],expected_signals=[])]
    output=[]
    for case in cases:
        entry={k:v for k,v in case.items() if k!='fields'}
        entry['prepared_sha256']=hashlib.sha256(prepare_text(case['fields']).encode()).hexdigest()
        entry['before']={m:old.predict(case['fields'],m) for m in ['neural_network','logistic_regression']}
        entry['after']={m:current.predict(case['fields'],m) for m in ['neural_network','logistic_regression','word_char_logistic','context_encoder'] if m in current.config['thresholds']}
        entry['review']=review_record(case['fields'])
        actual={s['code'] for s in entry['review']['signals']}
        entry['behavior_expectation_passed']=actual==set(case['expected_signals'])
        output.append(entry)
    save_json(OUT/'diagnostics.json',{'warning':'Handwritten and user-reported cases are diagnostics only, not training data or real-world accuracy evidence. Full original equipment ad and free-equipment ad were not supplied, so original .001/.036 scores cannot be reproduced exactly.', 'cases':output})
    save_json(OUT/'original_sensitivity.json',{'courier':ablation(old,courier['fields']),'equipment_paid':ablation(old,CASES[0]['fields']),'equipment_free':ablation(old,CASES[1]['fields'])})
    print(f'Diagnostic expectations passed: {sum(r["behavior_expectation_passed"] for r in output)}/{len(output)}')

if __name__=='__main__':main()
