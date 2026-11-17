"""Handwritten tests check software behavior, not real-world detection accuracy."""
import hashlib
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from shared.inference import Predictor
from shared.review import review_record,review_policy
from shared.text import clean_text,prepare_full_text
from diagnostics.cases import CASES

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_contextual_behavior_probes(case):
    result=review_record(case['fields'])
    assert result['method']=='heuristic'
    assert {s['code'] for s in result['signals']}==set(case['expected_signals'])
    for signal in result['signals']:
        assert signal['method']=='heuristic'
        for evidence in signal['evidence']:
            text=clean_text(case['fields'].get(evidence['field']))
            assert text[evidence['start']:evidence['end']]==evidence['text']

@pytest.fixture(scope='module')
def predictor():return Predictor()

def test_pair_holds_all_other_fields_constant(predictor):
    paid,free=CASES[0],CASES[1]
    assert set(k for k in paid['fields'] if paid['fields'][k]!=free['fields'][k])=={'requirements'}
    first=predictor.predict(paid['fields']);second=predictor.predict(free['fields'])
    assert first['review']['signals']
    assert not second['review']['signals']
    assert first['overall_review']['status']=='additional_review_recommended'
    # The review layer must not rewrite a learned score or class.
    assert first['model_score']==next(r for r in first['comparisons'] if r['model_name']=='neural_network')['model_score']
    assert not first['overall_review']['confirmed_illegal_activity']

def test_complete_input_and_shared_text(predictor,monkeypatch):
    record={'title':'Coordinator','company_profile':'Operations team','description':'A '*17000+'FINAL DESCRIPTION MARKER','requirements':'UNIQUE REQUIREMENTS END','benefits':'LAST BENEFIT SENTINEL'}
    observed=[]
    original=predictor.matrix_for
    def capture(texts,model):
        observed.append((model,list(texts)))
        return original(texts,model)
    monkeypatch.setattr(predictor,'matrix_for',capture)
    result=predictor.predict(record)
    full=prepare_full_text(record)
    assert observed==[('neural_network',[full]),('logistic_regression',[full])]
    assert full.endswith('LAST BENEFIT SENTINEL')
    assert result['input_summary']['prepared_sha256']==hashlib.sha256(full.encode()).hexdigest()
    assert result['input_summary']['truncated'] is False

def test_later_instructions_do_not_change_learned_scores(predictor):
    base=CASES[17]['fields'].copy()
    original=base.pop('later_instructions')
    a=predictor.predict(base);b=predictor.predict({**base,'later_instructions':original})
    assert a['comparisons']==b['comparisons']
    assert not a['review']['signals']
    assert b['review']['signals']

def test_boundary_rejected_instead_of_silently_clipped():
    with TestClient(app) as client:
        response=client.post('/api/v1/predictions',json={'title':'x','description':'y'*59999})
        assert response.status_code==422
        assert 'No content was silently removed' in response.json()['detail'][0]['msg']

def test_disagreement_does_not_combine_scores_or_confirm_crime():
    low={'model_name':'neural_network','predicted_label':0,'model_score':.236}
    high={'model_name':'logistic_regression','predicted_label':1,'model_score':.831}
    result=review_policy(low,[low,high],{'signals':[]})
    assert result['status']=='additional_review_recommended'
    assert result['models_disagree']
    assert result['confirmed_illegal_activity'] is False
    assert 'combined_probability' not in result

def test_review_reaches_end_of_long_unpunctuated_ad():
    text=('Maintain records and coordinate schedules '*400)+' Applicants must buy a laptop through our approved vendor before signing a contract. We reimburse the cost later.'
    assert 'candidate_funded_equipment' in {s['code'] for s in review_record({'description':text})['signals']}

def test_encoder_chunking_keeps_every_token_and_reload(predictor):
    if predictor.context_encoder is None:
        pytest.skip('Optional fine-tuned encoder is not in the active bundle')
    from shared.encoder import make_batch,CHUNK_SIZE
    tokenizer=predictor.encoder_tokenizer
    tokens=tokenizer.encode('Documented equipment provision and payment procedures. '*150,add_special_tokens=False,truncation=False)
    batch,owners,weights,count=make_batch([tokens],tokenizer)
    restored=[]
    for ids,mask in zip(batch['input_ids'],batch['attention_mask']):
        restored.extend(ids[1:int(mask.sum())-1].tolist())
    assert restored==tokens
    assert int(weights.sum())==len(tokens)
    assert count==1
    assert not predictor.context_encoder.training
    record={'description':'The employer provides a laptop. No payment is required.'}
    result=predictor.predict(record,'context_encoder')
    reloaded=Predictor().predict(record,'context_encoder')
    assert result==reloaded

def test_active_v2_contains_both_new_fitted_models(predictor):
    if predictor.config['model_version']!='emscad-v2':
        pytest.skip('Original v1 bundle selected')
    assert predictor.word_char_logistic is not None
    assert predictor.context_encoder is not None
    assert set(['neural_network','logistic_regression','word_char_logistic','context_encoder']).issubset(predictor.config['thresholds'])

def test_company_transactions_and_separate_payroll_are_distinct():
    record={'description':'Receive client payments in the company bank account. Transfer the funds to vendors from the company account. Your salary is paid into your personal bank account.'}
    assert not review_record(record)['signals']
