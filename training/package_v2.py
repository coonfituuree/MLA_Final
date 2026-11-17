"""Package validation-selected experiments without changing v1 or the fixed split."""
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from training.prepare import ROOT,save_json

VERSION='emscad-v2'

def main(preview=False):
    old=ROOT/'artifacts/emscad-v1';new=ROOT/'artifacts'/('experiments/integration-preview' if preview else VERSION)
    new.mkdir(parents=True,exist_ok=True)
    sparse=ROOT/'artifacts/experiments/sparse-v2'
    selection=json.loads((sparse/'selection.json').read_text())
    if selection['selected']['validation']['average_precision'] < selection['original_validation']['average_precision']:
        raise ValueError('Candidate did not improve validation AP; do not replace the original model.')
    for name in ['vectorizer.joblib','baseline.joblib','neural_network.pt','logistic_regression.joblib']:
        shutil.copyfile(old/name,new/name)
    shutil.copyfile(sparse/'features.joblib',new/'logistic_features.joblib')
    shutil.copyfile(sparse/'estimator.joblib',new/'word_char_logistic.joblib')
    config=json.loads((old/'config.json').read_text())
    config.update(model_version=VERSION,model_versions={'neural_network':'emscad-v1-mlp','logistic_regression':'emscad-v1-lr','word_char_logistic':'emscad-v2-word-char-lr'})
    config['thresholds']['word_char_logistic']=selection['selected']['validation']['threshold']
    config['training']['word_char_v2']=selection
    config['evaluation_scope']='Validation used for selection; original test already inspected in v1 and retained as a reused descriptive benchmark. No new independent generalization claim; illegal-recruitment labels unavailable.'
    encoder=ROOT/'artifacts/experiments/encoder-v2'
    if (encoder/'selection.json').exists():
        encoder_config=json.loads((encoder/'selection.json').read_text())
        shutil.copytree(encoder,new/'encoder',dirs_exist_ok=True)
        shutil.copyfile(ROOT/'data/evidence/BERT-APACHE-2.0-LICENSE.txt',new/'encoder/LICENSE.txt')
        shutil.copyfile(ROOT/'data/evidence/BERT-NOTICE.txt',new/'encoder/NOTICE.txt')
        config['context_encoder']=encoder_config
        config['model_versions']['context_encoder']='emscad-v2-tiny-bert'
        config['thresholds']['context_encoder']=encoder_config['selected']['validation']['threshold']
        for name in ['transformers','tokenizers','huggingface-hub','safetensors']:
            config['environment'][name]=importlib.metadata.version(name)
    save_json(new/'config.json',config)
    # Do not point the service at an incomplete bundle. Evaluation uses explicit path.
    from training.evaluate_v2 import evaluate
    evaluate(new)
    save_json(new/'manifest.json',{str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(new.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    from shared.inference import Predictor
    Predictor(new)
    if not preview:(ROOT/'artifacts/current.txt').write_text(VERSION+'\n')
    print('Verified preview bundle' if preview else 'Activated complete verified bundle',new)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    main(preview=parser.parse_args().preview)
