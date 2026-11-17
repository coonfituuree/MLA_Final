"""Run the full CPU experiment. Test labels are used only in final evaluation."""
import copy
import json
import random
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from shared.model import TfidfMLP, neural_scores
from shared.text import PREPROCESSING_VERSION
from training.prepare import prepare, save_json, ROOT, SEED

VERSION = 'emscad-v1'
BUNDLE = ROOT / 'artifacts' / VERSION

def threshold_for_f1(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)
    f = 2*p[:-1]*r[:-1] / np.maximum(p[:-1]+r[:-1], 1e-12)
    return float(thresholds[np.argmax(f)])

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / "manifest.json").unlink(missing_ok=True)
    data = prepare()
    masks = {p: data.partition.eq(p).to_numpy() for p in ['train', 'validation', 'test']}
    tfidf_config = dict(max_features=12000, min_df=2, max_df=.98, ngram_range=(1,2), sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    vectorizer = TfidfVectorizer(**tfidf_config)
    X = {'train': vectorizer.fit_transform(data.loc[masks['train'], 'text'])}
    for p in ['validation', 'test']:
        X[p] = vectorizer.transform(data.loc[masks[p], 'text'])
    y = {p: data.loc[masks[p], 'fraudulent'].to_numpy(dtype=np.int64) for p in masks}
    results = []
    best_lr, best_ap = None, -1
    for c in [.5, 2., 8.]:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=500, random_state=SEED, solver='liblinear')
        model.fit(X['train'], y['train'])
        score = model.predict_proba(X['validation'])[:,1]
        ap = average_precision_score(y['validation'], score)
        results.append(dict(model='logistic_regression', C=c, validation_ap=float(ap)))
        print(results[-1], flush=True)
        if ap > best_ap:
            best_lr, best_ap = model, ap
    lr_threshold = threshold_for_f1(y['validation'], best_lr.predict_proba(X['validation'])[:,1])
    positive_weight = float((y['train'] == 0).sum() / (y['train'] == 1).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    neural_trials = [dict(hidden_dim=64, dropout=.3, lr=.003), dict(hidden_dim=128, dropout=.5, lr=.001)]
    best_nn, best_nn_ap = None, -1
    for trial_index, trial in enumerate(neural_trials):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        arch = dict(input_dim=X['train'].shape[1], hidden_dim=trial['hidden_dim'], dropout=trial['dropout'])
        model = TfidfMLP(**arch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trial['lr'], weight_decay=.0001)
        history, best_state, trial_ap, stale, best_epoch = [], None, -1, 0, 0
        for epoch in range(1, 31):
            model.train()
            total_loss = 0.
            for indices in np.array_split(rng.permutation(len(y['train'])), int(np.ceil(len(y['train'])/128))):
                xb = torch.from_numpy(X['train'][indices].toarray())
                yb = torch.from_numpy(y['train'][indices].astype(np.float32))
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            scores = neural_scores(model, X['validation'])
            validation_total = 0.
            with torch.inference_mode():
                for start in range(0, len(y['validation']), 128):
                    xb = torch.from_numpy(X['validation'][start:start+128].toarray())
                    yb = torch.tensor(y['validation'][start:start+128], dtype=torch.float32)
                    validation_total += criterion(model(xb), yb).item() * len(yb)
            val_loss = validation_total / len(y['validation'])
            ap = float(average_precision_score(y['validation'], scores))
            history.append(dict(epoch=epoch, train_loss=total_loss/len(y['train']), validation_loss=val_loss, validation_ap=ap))
            if ap > trial_ap + .0001:
                best_state, trial_ap, stale, best_epoch = copy.deepcopy(model.state_dict()), ap, 0, epoch
            else:
                stale += 1
            print(f'NN trial {trial_index+1}, epoch {epoch}: train={history[-1]["train_loss"]:.4f} val={val_loss:.4f} AP={ap:.4f}', flush=True)
            if stale >= 5:
                break
        model.load_state_dict(best_state)
        results.append(dict(model='neural_network', **trial, validation_ap=trial_ap, best_epoch=best_epoch, epochs_run=len(history), history=history))
        if trial_ap > best_nn_ap:
            best_nn, best_nn_ap, best_arch = model, trial_ap, arch
            best_history, best_trial = history, results[-1]
    nn_threshold = threshold_for_f1(y['validation'], neural_scores(best_nn, X['validation']))
    baseline = DummyClassifier(strategy='most_frequent').fit(X['train'], y['train'])
    joblib.dump(vectorizer, BUNDLE / 'vectorizer.joblib')
    joblib.dump(best_lr, BUNDLE / 'logistic_regression.joblib')
    joblib.dump(baseline, BUNDLE / 'baseline.joblib')
    torch.save(best_nn.state_dict(), BUNDLE / 'neural_network.pt')
    save_json(ROOT / 'reports/search.json', results)
    save_json(ROOT / 'reports/neural_history.json', best_history)
    config = {
        'model_version': VERSION, 'preprocessing_version': PREPROCESSING_VERSION,
        'class_mapping': {'0':'legitimate', '1':'potentially_fraudulent'},
        'architecture': best_arch,
        'thresholds': {'neural_network':nn_threshold, 'logistic_regression':lr_threshold, 'majority_baseline':.5},
        'default_model':'neural_network', 'seed':SEED,
        'training': {'tfidf': {**tfidf_config, 'dtype':'float32'}, 'positive_weight':positive_weight,
            'optimizer':'AdamW', 'loss':'BCEWithLogitsLoss', 'batch_size':128, 'epoch_limit':30,
            'early_stopping':'validation average precision; patience 5; min_delta 0.0001',
            'weight_decay':.0001, 'selected_neural_trial':{k:v for k,v in best_trial.items() if k != 'history'},
            'selected_logistic_C': best_lr.C, 'logistic_class_weight':'balanced',
            'threshold_selection':'Maximum fraudulent-class F1 on validation scores; first (lowest) threshold on ties.',
            'selection_metric':'validation average precision', 'split':'5-fold StratifiedGroupKFold(seed=42); fold 0 test, fold 1 validation, folds 2-4 train. Long identical employer profiles/descriptions and verified near-duplicates grouped.'},
        'provenance':json.loads((ROOT/'data/provenance.json').read_text()),
        'environment':{'python':platform.python_version(), **{name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','joblib']}},
        'scope':'English EMSCAD advertisements from 2012–2014; historical platform and geography biases. Scores are uncalibrated screening signals.'
    }
    save_json(BUNDLE / 'config.json', config)
    # Verify a round trip before evaluating the test partition.
    reloaded = TfidfMLP(**best_arch)
    reloaded.load_state_dict(torch.load(BUNDLE/'neural_network.pt', map_location='cpu', weights_only=True))
    np.testing.assert_allclose(neural_scores(best_nn, X['validation']), neural_scores(reloaded, X['validation']), atol=1e-7)
    assert np.allclose(joblib.load(BUNDLE/'logistic_regression.joblib').predict_proba(X['validation']), best_lr.predict_proba(X['validation']))
    (ROOT/'artifacts/current.txt').write_text(VERSION+'\n')
    from training.evaluate import evaluate
    evaluate(require_complete=False)

if __name__ == '__main__':
    main()
