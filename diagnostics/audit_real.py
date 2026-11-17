"""Frozen-rule audit on real EMSCAD validation advertisements.

Dataset fraud labels are not risk-specific gold labels. This reports warnings and
label overlap, not a fabricated estimate of equipment/money-mule detection accuracy.
"""
import hashlib
import json
import re
import pandas as pd
from shared.inference import ROOT
from shared.review import review_record,REVIEW_VERSION
from shared.text import FIELDS,prepare_text,clean_text
from training.prepare import save_json
from training.frozen_data import load_frozen_data

OUT=ROOT/'reports/improvements'

def redact(text):
    text=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL REDACTED]', text)
    text=re.sub(r'(?:https?://|www\.)\S+', '[URL REDACTED]', text)
    text=re.sub(r'(?<!\w)\+?\d[\d ()-]{8,}\d', '[CONTACT NUMBER REDACTED]', text)
    return text

def main():
    rules=ROOT/'shared/review.py';digest=hashlib.sha256(rules.read_bytes()).hexdigest()
    lock=OUT/f'heuristic-audit-lock-{REVIEW_VERSION}.json'
    if lock.exists():
        if json.loads(lock.read_text())['rules_sha256']!=digest:
            raise ValueError('Rules changed after the real-data audit. Version the rules and disclose evaluation reuse before auditing again.')
    else:save_json(lock,{'version':REVIEW_VERSION,'rules_sha256':digest,'partition':'original_validation','labels':'EMSCAD fraud labels only; no independently adjudicated risk-specific labels','rule_development_data':'handwritten behavior probes and cited FTC/FBI guidance; these validation advertisements were not inspected to tune the rules.'})
    save_json(OUT/'heuristic-audit-lock.json',json.loads(lock.read_text()))
    split=pd.read_csv(ROOT/'reports/splits.csv');data,split=load_frozen_data()
    rows=data.iloc[split.loc[split.partition=='validation','row_index']]
    warnings=[];counts={};topic_counts={}
    for _,row in rows.iterrows():
        record=row.to_dict();review=review_record(record)
        for topic,pattern in [('courier',r'\bcourier\b|document delivery'),('remote',r'work from home|\bremote\b'),('reimbursement',r'\breimburs'),('payments',r'payment processing|process payments|accounts payable'),('equipment',r'\bequipment\b')]:
            if re.search(pattern,prepare_text(record),re.I):
                entry=topic_counts.setdefault(topic,{'count':0,'dataset_legitimate':0,'dataset_fraudulent':0,'warnings':0,'warnings_on_dataset_legitimate':0})
                entry['count']+=1;entry['dataset_legitimate' if int(row.fraudulent)==0 else 'dataset_fraudulent']+=1
                entry['warnings']+=int(bool(review['signals']))
                entry['warnings_on_dataset_legitimate']+=int(bool(review['signals']) and int(row.fraudulent)==0)
        if review['signals']:
            for signal in review['signals']:counts[signal['code']]=counts.get(signal['code'],0)+1
            # Store identifiers and redacted quotations, never whole ads/contact fields.
            for signal in review['signals']:
                for evidence in signal['evidence']:
                    evidence['text']=redact(evidence['text'])
                    evidence['note']='Redacted for the report; offsets refer to original normalized field text.'
            warnings.append({'job_id':int(row.job_id),'dataset_fraud_label':int(row.fraudulent),'source_type':'advertisement','source':'EMSCAD CSV SHA-256 in data/provenance.json','review':review})
    save_json(OUT/'heuristic_real_audit.json',{'rules_sha256':digest,'version':REVIEW_VERSION,'partition':'validation','total':len(rows),'warning_ads':len(warnings),'warnings_on_dataset_legitimate':sum(x['dataset_fraud_label']==0 for x in warnings),'warnings_on_dataset_fraudulent':sum(x['dataset_fraud_label']==1 for x in warnings),'signals':counts,'topic_counts':topic_counts,'cases':warnings,'interpretation':'Warning overlap with EMSCAD labels is descriptive. There are no independently verified labels for these specific risk patterns. Zero warnings/false alarms in a stratum is not evidence of risk recall or employer verification.'})
    # Real negative controls from already inspected TRAINING ads. No tuning or formal accuracy claim.
    controls=[(50,'courier','Provide courier services'),(3781,'reimbursement','Generous reimbursement policy for job-related IT equipment.'),(11,'remote','Desire and ability to work from home'),(33,'entry_level','Entry-Level Craftsman Associate Position'),(4914,'employer_provided_equipment','A new laptop to work on is provided to each new employee.'),(964,'administrative_payments','Will prepare and mail checks.')]
    records=[]
    for job_id,topic,quote in controls:
        row=data.loc[data.job_id==job_id].iloc[0];text=prepare_text(row.to_dict());assert quote in text
        assert int(row.fraudulent)==0
        records.append({'job_id':job_id,'topic':topic,'source_type':'real_EMSCAD_advertisement','source_partition':'training','dataset_fraud_label':0,'evidence_status':'historical_dataset_label_not_independent_present_day_verification','excerpt':quote,'excerpt_sha256':hashlib.sha256(quote.encode()).hexdigest(),'review_signal_codes':[s['code'] for s in review_record(row.to_dict())['signals']],'use':'sourced qualitative negative control; not independent accuracy evidence'})
    save_json(ROOT/'data/evidence/emscad_controls.json',records)
    print('Real-data heuristic audit:',len(warnings),'warnings across',len(rows),'validation advertisements')

if __name__=='__main__':main()
