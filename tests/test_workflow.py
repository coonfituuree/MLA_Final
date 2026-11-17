"""Meaningful checks for shared preprocessing, trusted artifacts, and HTTP contracts."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from shared.inference import Predictor, ROOT
from shared.text import prepare_text, clean_text

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope='module')
def heldout():
    split = pd.read_csv(ROOT/'reports/splits.csv')
    row_index = split.loc[split.partition.eq('test'), 'row_index'].iloc[0]
    row = pd.read_csv(ROOT/'data/raw/fake_job_postings.csv').iloc[row_index].fillna('')
    return {f: row[f] for f in ['title','company_profile','description','requirements','benefits']}

def test_preparation_handles_html_and_missing_fields():
    assert prepare_text({'title':None,'description':'<p>Hello&nbsp; world</p><script>hidden()</script><p>Apply now.</p>'}) == 'Hello world Apply now.'
    assert prepare_text({'description':'a\n b', 'fraudulent':1,'job_id':999}) == 'a b'
    assert clean_text(float('nan')) == ''

@pytest.mark.parametrize('model',['neural_network','logistic_regression','word_char_logistic','context_encoder'])
def test_artifact_round_trip_and_http_consistency(client,heldout,model):
    first, second = Predictor(), Predictor()
    if model not in first.config['thresholds']:pytest.skip('Optional fitted model not in active bundle')
    a = first.predict(heldout,model)
    b = second.predict(heldout,model)
    assert a == b
    response = client.post('/api/v1/predictions',json={**heldout,'model':model})
    assert response.status_code == 200
    assert response.json() == a
    assert not first.neural.training
    assert first.device == 'cpu'

def test_ready_and_real_metrics(client):
    assert client.get('/api/health').json()['status'] == 'ok'
    assert client.get('/api/ready').status_code == 200
    info=client.get('/api/v1/model-info').json()
    assert info['metrics']['neural_network']['test_count'] > 1000
    assert info['metrics']['majority_baseline']['recall'] == 0

@pytest.mark.parametrize('body',[{}, {'description':''}, {'description':'   '}, {'description':'<p> </p>'}, {'description':'<script>x()</script>'}, {'description':'text','fraudulent':1}, {'description':'text','model':'imaginary'}, {'description':'x'*60001}])
def test_invalid_json(client,body):
    assert client.post('/api/v1/predictions',json=body).status_code == 422

@pytest.mark.parametrize('name,content,status',[('ad.txt',b'',422),('ad.txt',b'\xff\xfe',422),('ad.pdf',b'content',422),('ad.txt',b'x'*120001,413),('ad.txt',b'\x00hello',422)])
def test_invalid_uploads(client,name,content,status):
    assert client.post('/api/v1/predictions/file',files={'file':(name,content)}).status_code == status

def test_file_uses_same_pipeline(client,heldout):
    text = heldout['description']
    expected=client.post('/api/v1/predictions',json={'description':text}).json()
    actual=client.post('/api/v1/predictions/file',files={'file':('vacancy.txt',text.encode('utf-8'),'text/plain')})
    assert actual.status_code == 200
    assert actual.json() == expected
    assert client.post('/api/v1/predictions/file',files={'file':('a.txt',text.encode())},data={'model':'unknown'}).status_code==422

def test_missing_artifacts_are_actionable(monkeypatch,tmp_path):
    monkeypatch.setenv('MODEL_BUNDLE',str(tmp_path/'absent'))
    with TestClient(app) as missing:
        assert missing.get('/api/health').status_code == 200
        assert missing.get('/api/ready').status_code == 503
        result=missing.post('/api/v1/predictions',json={'description':'A real description'})
        assert result.status_code == 503
        assert 'training.train' in result.json()['detail']

def test_partition_integrity_and_train_only_vocabulary():
    splits=pd.read_csv(ROOT/'reports/splits.csv')
    assert splits.groupby('group').partition.nunique().max()==1
    assert splits.row_index.is_unique
    assert len(splits)==15872
    predictor=Predictor()
    # A held-out term absent from training must not leak into the fitted vocabulary.
    data=pd.read_csv(ROOT/'data/raw/fake_job_postings.csv').fillna('')
    analyzer=predictor.vectorizer.build_analyzer()
    train_terms=set()
    for row in data.iloc[splits.loc[splits.partition.eq('train'),'row_index']].to_dict('records'):
        train_terms.update(analyzer(prepare_text(row)))
    assert set(predictor.vectorizer.vocabulary_).issubset(train_terms)

def test_corrupt_artifact_rejected(tmp_path):
    predictor=Predictor()
    (tmp_path/'manifest.json').write_text(json.dumps({'neural_network.pt':'incorrect'}))
    (tmp_path/'neural_network.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum mismatch'):
        Predictor(tmp_path)

def test_body_size_limit(client):
    assert client.post('/api/v1/predictions',content=b'x'*300001).status_code==413
    chunks=(b'x'*100000 for _ in range(4))
    assert client.post('/api/v1/predictions',content=chunks).status_code==413
