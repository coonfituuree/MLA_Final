"""Create a real held-out browser fixture without using it for model selection."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from shared.inference import Predictor, ROOT

def main():
    split = pd.read_csv(ROOT/'reports/splits.csv')
    data = pd.read_csv(ROOT/'data/raw/fake_job_postings.csv').fillna('')
    index = split.loc[(split.partition=='test') & (split.fraudulent==1), 'row_index'].iloc[0]
    row = data.iloc[index]
    text = row.description
    predictor = Predictor()
    fixture = {'source':'EMSCAD official Kaggle CSV, first fraudulent test row in original order; description-only input',
               'job_id':int(row.job_id), 'description':text,
               'neural_score':predictor.predict({'description':text})['model_score'],
               'logistic_score':predictor.predict({'description':text},'logistic_regression')['model_score']}
    fixture['optional_model_scores']={name:predictor.predict({'description':text},name)['model_score'] for name in ['word_char_logistic','context_encoder'] if name in predictor.config['thresholds']}
    (ROOT/'reports/browser-fixture.json').write_text(json.dumps(fixture,indent=2))
    print(f'Browser fixture: real test job_id={int(row.job_id)}')

if __name__=='__main__':
    main()
