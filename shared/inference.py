"""Load trusted artifacts once; prepare one complete ad for all learned classifiers."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import joblib
import torch
from shared.model import TfidfMLP, neural_scores
from shared.text import prepare_text, prepare_full_text, FIELDS, MAX_TEXT, clean_text, PREPROCESSING_VERSION
from shared.review import review_record, review_policy

ROOT = Path(__file__).resolve().parents[1]

class Predictor:
    def __init__(self, bundle=None, require_complete=True):
        if bundle is None:
            configured = os.environ.get('MODEL_BUNDLE')
            bundle = Path(configured).expanduser().resolve() if configured else ROOT/'artifacts'/(ROOT/'artifacts/current.txt').read_text().strip()
        self.bundle = Path(bundle).resolve()
        self.device = os.environ.get('MODEL_DEVICE', 'cpu')
        torch.set_num_threads(2)
        if require_complete:
            manifest = json.loads((self.bundle/'manifest.json').read_text())
            for filename, digest in manifest.items():
                # Nested tokenizer files are allowed only within this trusted bundle.
                path=(self.bundle/filename).resolve()
                if not path.is_relative_to(self.bundle):
                    raise ValueError('Invalid artifact manifest filename')
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Artifact checksum mismatch: {filename}. Retrain or restore the trusted bundle.')
        self.config = json.loads((self.bundle/'config.json').read_text())
        if self.config['preprocessing_version'] != PREPROCESSING_VERSION:
            raise ValueError('Preprocessing version mismatch; retrain with this source code.')
        for name in ['torch','scikit-learn','numpy','scipy','joblib']:
            if importlib.metadata.version(name) != self.config['environment'][name]:
                raise ValueError(f'{name} version mismatch. Install requirements.lock.txt or retrain.')
        self.vectorizer = joblib.load(self.bundle/'vectorizer.joblib')
        self.logistic_features = joblib.load(self.bundle/'logistic_features.joblib') if (self.bundle/'logistic_features.joblib').exists() else self.vectorizer
        self.logistic = joblib.load(self.bundle/'logistic_regression.joblib')
        self.word_char_logistic=joblib.load(self.bundle/'word_char_logistic.joblib') if (self.bundle/'word_char_logistic.joblib').exists() else None
        self.baseline = joblib.load(self.bundle/'baseline.joblib')
        self.neural = TfidfMLP(**self.config['architecture']).to(self.device)
        self.neural.load_state_dict(torch.load(self.bundle/'neural_network.pt', map_location=self.device, weights_only=True))
        self.neural.eval()
        self.context_encoder=None
        if self.config.get('context_encoder'):
            from shared.encoder import ChunkedEncoder
            from transformers import BertTokenizerFast
            for name in ['transformers','tokenizers']:
                if importlib.metadata.version(name) != self.config['environment'][name]:
                    raise ValueError(f'{name} version mismatch. Install requirements.lock.txt.')
            source=self.bundle/'encoder/tokenizer'
            self.encoder_tokenizer=BertTokenizerFast.from_pretrained(source,local_files_only=True)
            self.context_encoder=ChunkedEncoder(source).to(self.device)
            self.context_encoder.load_state_dict(torch.load(self.bundle/'encoder/encoder.pt',map_location=self.device,weights_only=True))
            self.context_encoder.eval()
        self.metrics = json.loads((self.bundle/'metrics.json').read_text()) if (self.bundle/'metrics.json').exists() else {}

    def matrix_for(self,texts,model):
        if model=='context_encoder':
            from shared.encoder import encode_documents
            return encode_documents(self.encoder_tokenizer,texts)
        return (self.logistic_features if model=='word_char_logistic' else self.vectorizer).transform(texts)

    def scores(self, matrix, model):
        if model == 'neural_network':return neural_scores(self.neural, matrix, self.device)
        if model == 'logistic_regression':return self.logistic.predict_proba(matrix)[:,1]
        if model == 'word_char_logistic' and self.word_char_logistic is not None:return self.word_char_logistic.predict_proba(matrix)[:,1]
        if model == 'majority_baseline':return self.baseline.predict_proba(matrix)[:,1]
        if model == 'context_encoder' and self.context_encoder is not None:
            from shared.encoder import encoder_scores
            return encoder_scores(self.context_encoder,self.encoder_tokenizer,matrix,device=self.device)
        raise ValueError('Unknown or unavailable model')

    def _prediction(self,text,model):
        matrix=self.matrix_for([text],model)
        score=float(self.scores(matrix,model)[0])
        threshold=self.config['thresholds'][model]
        label=int(score>=threshold)
        return dict(predicted_label=label,label=self.config['class_mapping'][str(label)],model_score=score,
                    decision_threshold=threshold,model_name=model,
                    model_version=self.config.get('model_versions',{}).get(model,self.config['model_version']),
                    score_note='Uncalibrated learned fraud score; not a verified probability or a finding of criminal activity.',
                    vocabulary_match=bool(matrix.nnz>0) if model!='context_encoder' else bool(matrix[0]))

    def predict(self,record,model='neural_network'):
        full=prepare_full_text(record)
        if not full:raise ValueError('Provide readable vacancy text after HTML removal.')
        if len(full)>MAX_TEXT:raise ValueError('Prepared advertisement exceeds 60,000 characters. Shorten it; inference does not silently truncate.')
        if model not in self.config['thresholds'] or model=='majority_baseline':
            raise ValueError('This model is unavailable in the loaded bundle. Select an available model from model-info.')
        # One immutable prepared string is shared by both primary models.
        comparisons=[self._prediction(full,m) for m in ['neural_network','logistic_regression']]
        selected=next((r for r in comparisons if r['model_name']==model),None)
        if selected is None:
            selected=self._prediction(full,model);comparisons.append(selected)
        review=review_record(record)
        return {**selected,'comparisons':comparisons,'review':review,'overall_review':review_policy(selected,comparisons,review),
                'input_summary':{'prepared_characters':len(full),'prepared_sha256':hashlib.sha256(full.encode()).hexdigest(),
                                 'included_fields':[f for f in FIELDS if clean_text(record.get(f))],
                                 'preprocessing_version':PREPROCESSING_VERSION,'truncated':False,'later_instructions_in_learned_input':False},
                'bundle_version':self.config['model_version']}
