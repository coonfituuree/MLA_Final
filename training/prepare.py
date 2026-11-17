"""Inspect EMSCAD and create deterministic, duplicate- and employer-grouped partitions."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.text import FIELDS, prepare_text, clean_text

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))

def prepare():
    source = ROOT / 'data/raw/fake_job_postings.csv'
    if not source.exists():
        raise SystemExit('Missing dataset. Run python scripts/download_data.py or place CSV in data/raw/.')
    provenance = json.loads((ROOT / 'data/provenance.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['sha256']:
        raise ValueError('Dataset checksum differs from recorded Kaggle source. Restore the CSV before training.')
    df = pd.read_csv(source)
    assert set(FIELDS + ('fraudulent',)).issubset(df.columns)
    assert set(df.fraudulent.unique()) == {0, 1}
    texts = df.apply(prepare_text, axis=1)
    normalized = texts.str.casefold()
    hashes = normalized.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    conflict_hashes = set(df.assign(h=hashes).groupby('h').fraudulent.nunique().loc[lambda x: x > 1].index)
    conflict = hashes.isin(conflict_hashes)
    empty = texts.str.len().eq(0)
    duplicate = hashes.duplicated(keep='first')
    usable = df.loc[~(conflict | empty | duplicate)].copy()
    usable['row_index'] = usable.index
    usable['text'] = texts.loc[usable.index]
    usable = usable.reset_index(drop=True)
    n = len(usable)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    # Exact long profiles proxy shared employers; exact descriptions capture reposts
    # despite changes to title/benefits. Neither uses labels.
    for field, minimum in [('company_profile', 150), ('description', 100)]:
        seen = {}
        for i, value in enumerate(usable[field]):
            key = clean_text(value).casefold()
            if len(key) >= minimum:
                if key in seen:
                    union(i, seen[key])
                else:
                    seen[key] = i
    lsh = MinHashLSH(threshold=0.75, num_perm=64)
    shingles = []
    near_pairs = []
    print(f'Grouping {n} usable advertisements with token-shingle MinHash...', flush=True)
    for i, text in enumerate(usable.text):
        tokens = re.findall(r'\w+', text.casefold())
        token_set = {' '.join(tokens[j:j+5]) for j in range(max(0, len(tokens)-4))}
        shingles.append(token_set)
        if len(token_set) < 15:
            continue
        signature = MinHash(num_perm=64, seed=SEED)
        signature.update_batch([s.encode() for s in sorted(token_set)])
        for key in lsh.query(signature):
            j = int(key)
            other = shingles[j]
            similarity = len(token_set & other) / len(token_set | other)
            if similarity >= 0.85:
                union(i, j)
                near_pairs.append((i, j, similarity))
        lsh.insert(str(i), signature)
        if i % 3000 == 0:
            print(f'Grouped {i}/{n}', flush=True)
    groups = np.array([find(i) for i in range(n)])
    folds = np.zeros(n, dtype=int)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, holdout) in enumerate(splitter.split(usable.text, usable.fraudulent, groups)):
        folds[holdout] = fold
    partitions = np.where(folds == 0, 'test', np.where(folds == 1, 'validation', 'train'))
    usable['group'] = groups
    usable['partition'] = partitions
    assert not any(partitions[i] != partitions[j] for i, j, _ in near_pairs)
    for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
        assert not set(groups[partitions == a]) & set(groups[partitions == b])
    usable[['row_index', 'job_id', 'fraudulent', 'group', 'partition']].to_csv(ROOT / 'reports/splits.csv', index=False)
    df.loc[conflict, ['job_id', 'fraudulent']].to_csv(ROOT / 'reports/quarantined.csv', index=False)
    lengths = texts.str.split().str.len()
    eda = {
        'rows': len(df), 'columns': len(df.columns), 'class_counts': df.fraudulent.value_counts().sort_index().to_dict(),
        'missing': df.isna().sum().to_dict(), 'empty_clean_fields': {f: int(df[f].map(clean_text).eq('').sum()) for f in FIELDS},
        'exact_full_rows_excluding_id': int(df.drop(columns='job_id').duplicated().sum()),
        'normalized_duplicate_excess': int(duplicate.sum()), 'conflicting_groups': len(conflict_hashes),
        'quarantined_rows': int(conflict.sum()), 'empty_combined_rows': int(empty.sum()), 'usable_rows': n,
        'word_lengths_by_class': {str(label): lengths[df.fraudulent == label].describe(percentiles=[.1,.5,.9,.99]).to_dict() for label in [0, 1]},
        'missing_profile_fraud_rate': float(df.loc[df.company_profile.isna(), 'fraudulent'].mean()),
        'present_profile_fraud_rate': float(df.loc[df.company_profile.notna(), 'fraudulent'].mean()),
        'countries': df.location.fillna('Unknown').str.split(',').str[0].value_counts().head(12).to_dict(),
        'groups': len(set(groups)), 'largest_group': max(Counter(groups).values()),
        'near_duplicate_pairs_verified': len(near_pairs), 'cross_split_verified_near_pairs': 0,
        'near_duplicate_method': '64-permutation MinHash LSH candidates (0.75), exact 5-word-shingle Jaccard >=0.85; approximate candidate search, not exhaustive.',
        'split_counts': usable.groupby(['partition', 'fraudulent']).size().unstack(fill_value=0).to_dict(orient='index'),
    }
    save_json(ROOT / 'reports/eda.json', eda)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df.fraudulent.value_counts().sort_index().plot.bar(ax=axes[0,0], color=['#2f776c','#c97b45'])
    axes[0,0].set(title='Class imbalance: original EMSCAD', xticklabels=['Legitimate', 'Fraudulent'], ylabel='Advertisements')
    for label, name in [(0, 'Legitimate'), (1, 'Fraudulent')]:
        axes[0,1].hist(lengths[df.fraudulent == label], bins=np.logspace(0,4,45), alpha=.6, density=True, label=name)
    axes[0,1].set(xscale='log', title='Combined text length', xlabel='Words (log scale)', ylabel='Density')
    axes[0,1].legend()
    pd.Series(eda['empty_clean_fields']).plot.barh(ax=axes[1,0], color='#2f776c')
    axes[1,0].set(title='Empty fields after HTML removal', xlabel='Advertisements')
    pd.Series(eda['countries']).plot.bar(ax=axes[1,1], color='#647b92')
    axes[1,1].set(title='Most frequent location country codes', ylabel='Advertisements')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/figures/eda.png', dpi=150)
    plt.close(fig)
    print(json.dumps(eda['split_counts']), flush=True)
    return usable

if __name__ == '__main__':
    prepare()
