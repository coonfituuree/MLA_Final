"""Fine-tune all tiny-BERT parameters on real EMSCAD training labels, CPU by default."""
import copy
import hashlib
import json
import random
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast
from shared.encoder import ChunkedEncoder, encode_documents, make_batch, encoder_scores, CHUNK_SIZE
from shared.text import prepare_text
from training.prepare import ROOT, save_json
from training.frozen_data import load_frozen_data
from training.train import threshold_for_f1
from training.experiments import metrics

OUT=ROOT/'artifacts/experiments/encoder-v2'
MODEL_ID='google/bert_uncased_L-2_H-128_A-2'
REVISION='30b0a37ccaaa32f332884b96992754e246e48c5f'

def main():
    from huggingface_hub import snapshot_download
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    source=snapshot_download(MODEL_ID,revision=REVISION,allow_patterns=['config.json','vocab.txt','model.safetensors','pytorch_model.bin','README.md'])
    tokenizer=BertTokenizerFast.from_pretrained(source,local_files_only=True)
    split=pd.read_csv(ROOT/'reports/splits.csv')
    assert (ROOT/'reports/splits.csv').read_bytes()==(ROOT/'reports/improvements/frozen_splits.csv').read_bytes()
    data,split=load_frozen_data()
    rows={p:data.iloc[split.loc[split.partition==p,'row_index']] for p in ['train','validation']}
    documents={p:encode_documents(tokenizer,rows[p].apply(prepare_text,axis=1).tolist()) for p in rows}
    y={p:rows[p].fraudulent.to_numpy(dtype=np.float32) for p in rows}
    weight=float((y['train']==0).sum()/(y['train']==1).sum())
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    trials=[];overall=None
    # One predeclared encoder setting, capped at three full-data CPU epochs.
    # The broader eight-run sparse search supplies the practical hyperparameter comparison.
    for lr in [2e-4]:
        torch.manual_seed(42);rng=np.random.default_rng(42)
        model=ChunkedEncoder(source,pretrained=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        best_ap=-1;best_state=None;stale=0;history=[]
        for epoch in range(1,4):
            model.train();total=0.
            order=rng.permutation(len(y['train']))
            for offset in range(0,len(order),8):
                ids=order[offset:offset+8]
                batch=make_batch([documents['train'][i] for i in ids],tokenizer)
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(*batch),torch.from_numpy(y['train'][ids]));loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();total+=loss.item()*len(ids)
                if offset % 2048 == 0:print(f'encoder lr={lr} epoch={epoch} documents={offset}/{len(order)}',flush=True)
            scores=encoder_scores(model,tokenizer,documents['validation'])
            threshold=threshold_for_f1(y['validation'],scores)
            result=metrics(y['validation'],scores,threshold)
            history.append(dict(epoch=epoch,train_loss=total/len(order),**result))
            print(f'encoder lr={lr} epoch={epoch} {history[-1]}',flush=True)
            if result['average_precision']>best_ap+.0001:
                best_ap=result['average_precision'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=2:break
        model.load_state_dict(best_state)
        scores=encoder_scores(model,tokenizer,documents['validation']);threshold=threshold_for_f1(y['validation'],scores)
        trial=dict(lr=lr,best_epoch=best_epoch,history=history,validation=metrics(y['validation'],scores,threshold))
        trials.append(trial)
        if overall is None or best_ap>overall['validation']['average_precision']:
            overall=trial;torch.save(best_state,OUT/'encoder.pt')
    shutil.copyfile(ROOT/'data/evidence/BERT-APACHE-2.0-LICENSE.txt',OUT/'LICENSE.txt')
    shutil.copyfile(ROOT/'data/evidence/BERT-NOTICE.txt',OUT/'NOTICE.txt')
    tokenizer.save_pretrained(OUT/'tokenizer')
    shutil.copyfile(source+'/config.json',OUT/'tokenizer/config.json')
    save_json(OUT/'selection.json',dict(selected=overall,trials=trials,source={'model_id':MODEL_ID,'revision':REVISION,'license':'Apache-2.0','url':'https://huggingface.co/'+MODEL_ID},architecture={'chunk_tokens':CHUNK_SIZE,'chunk_overlap':0,'pooling':'token-count weighted mean of within-chunk masked mean hidden states','dropout':.2,'all_input_tokens_retained':True},training={'seed':42,'optimizer':'AdamW','loss':'BCEWithLogitsLoss','positive_weight':weight,'batch_documents':8,'epoch_limit':3,'patience':2,'weight_decay':.01,'gradient_clip':1.,'fine_tuned_encoder_parameters':True,'train_count':len(y['train'])},split_sha256=hashlib.sha256((ROOT/'reports/splits.csv').read_bytes()).hexdigest(),test_used_for_selection=False))
    save_json(ROOT/'reports/improvements/encoder_search.json',json.loads((OUT/'selection.json').read_text()))

if __name__=='__main__':main()
