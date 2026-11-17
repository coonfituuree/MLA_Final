"""Integrity checks shared by follow-up experiments: never regenerate the split."""
import hashlib
import json
import pandas as pd
from shared.inference import ROOT

SPLIT_SHA256='4abc3a4449456421743d3f7a2dac6e69ff54e1af31ddefa5c1c8afce7f2a2b7a'

def load_frozen_data():
    source=ROOT/'data/raw/fake_job_postings.csv'
    provenance=json.loads((ROOT/'data/provenance.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance['sha256']:
        raise ValueError('Dataset checksum differs from the recorded real EMSCAD source.')
    split=ROOT/'reports/splits.csv'
    if hashlib.sha256(split.read_bytes()).hexdigest()!=SPLIT_SHA256:
        raise ValueError('Original split changed. Restore the fixed split before running follow-up experiments.')
    return pd.read_csv(source).fillna(''),pd.read_csv(split)
