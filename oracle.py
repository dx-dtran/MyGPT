import torch
import torch.nn as nn
import torch.nn.functional as F

# transformer
batch_size = 8
block_size = 8
n_embd = 16
n_head = 4
n_layer = 4
dropout = 0.0

# training
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cpu'

torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.keys_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.values_matrix = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        batch_dim, time_dim, features_dim = x.shape

        queries = self.query_matrix(x)  # (B, T, C) * (C, H) == (B, T, H)
        keys = self.keys_matrix(x)  # (B, T, C) * (C, H) == (B, T, H)

        out = queries.bmm(keys.transpose(1, 2)) ** (-1 / features_dim)  # (B, T, H) * (B, H, T) == (B, T, T)
        out = F.softmax(out, dim=1)

        values = self.keys_matrix(x)  # (B, T, C)

        return out.bmm(values)  # (B, T, T) * (B, T, C) == (B, T, C)


if __name__ == '__main__':
    pass
