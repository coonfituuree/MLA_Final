"""Fine-tunable compact BERT with all-text chunking and document-level pooling.

Every token is represented; individual chunks use local attention, not cross-chunk
attention. Token-count weighting prevents short final chunks dominating the mean.
"""
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, BertTokenizerFast

CHUNK_SIZE=192

def encode_documents(tokenizer,texts):
    return [tokenizer.encode(text,add_special_tokens=False,truncation=False) for text in texts]

def make_batch(documents,tokenizer,device='cpu'):
    chunks=[];owners=[];weights=[]
    for owner,tokens in enumerate(documents):
        for start in range(0,max(1,len(tokens)),CHUNK_SIZE):
            chunk=tokens[start:start+CHUNK_SIZE]
            chunks.append(tokenizer.build_inputs_with_special_tokens(chunk))
            owners.append(owner);weights.append(max(1,len(chunk)))
    batch=tokenizer.pad({'input_ids':chunks},padding=True,return_tensors='pt')
    return ({k:v.to(device) for k,v in batch.items()},torch.tensor(owners,device=device),torch.tensor(weights,dtype=torch.float32,device=device),len(documents))

class ChunkedEncoder(nn.Module):
    def __init__(self,source,pretrained=False,dropout=.2):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(source,local_files_only=True) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(source,local_files_only=True))
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(self.encoder.config.hidden_size,1)
    def forward(self,batch,owners,weights,count):
        hidden=self.encoder(**batch).last_hidden_state
        mask=batch['attention_mask'].unsqueeze(-1)
        representations=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        pooled=torch.zeros(count,representations.shape[1],device=hidden.device)
        totals=torch.zeros(count,device=hidden.device)
        pooled.index_add_(0,owners,representations*weights.unsqueeze(1))
        totals.index_add_(0,owners,weights)
        return self.classifier(self.dropout(pooled/totals.clamp_min(1).unsqueeze(1))).squeeze(1)

def encoder_scores(model,tokenizer,documents,batch_size=8,device='cpu'):
    import numpy as np
    model.eval();scores=[]
    with torch.inference_mode():
        for start in range(0,len(documents),batch_size):
            scores.extend(torch.sigmoid(model(*make_batch(documents[start:start+batch_size],tokenizer,device))).cpu().tolist())
    return np.asarray(scores)
