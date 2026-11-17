"""Sparse TF-IDF is densified only one minibatch at a time."""
import numpy as np
import torch
from torch import nn

class TfidfMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.network(x).squeeze(1)

def neural_scores(model, matrix, device='cpu', batch_size=128):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            x = torch.from_numpy(matrix[start:start + batch_size].toarray().astype(np.float32)).to(device)
            result.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(result)
