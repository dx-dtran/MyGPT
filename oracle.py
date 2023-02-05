import torch
import torch.nn as nn
from torch.nn import functional as F

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
        self.head_size = head_size
        self.key_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.query_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.value_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, time_dim, _ = x.shape

        # order: the input tensor multiplied by the linear layer
        keys = self.key_matrix(x)  # (B, T, C) @ (C, H) -> (B, T, H)
        queries = self.query_matrix(x)  # (B, T, C) @ (C, H) -> (B, T, H)

        keys = keys.transpose(2, 1)  # (B, H, T)
        out = queries.bmm(keys)  # (B, H, T) @ (B, T, H) -> (B, T, T)
        out = out / (self.head_size ** 0.5)

        # how can i initialize tril here in the forward pass without requiring grad?
        # kind of weird to initialize it in the init method with a block_size shape, when we only need it to be
        # time_dim shape
        out = torch.masked_fill(out, self.tril[:time_dim, :time_dim] == 0, float('-inf'))
        out = F.softmax(out, dim=2)  # (B, T, T)

        values = self.value_matrix(x)  # (B, T, C) @ (C, H) -> (B, T, H)

        # order: output tensor multiplied by the values tensor
        return out.bmm(values)  # (B, T, T) @ (B, T, H) -> (B, T, H)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        heads = [head(x) for head in self.heads]  # (B, T, H) each
        concatenated = torch.cat(heads, dim=2)  # (B, T, H * num_heads)
        out = self.linear(concatenated)  # (B, T, H * num_heads) @ (H * num_heads, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd * 4, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_embd * 4, n_embd, bias=True)

    def forward(self, x):
        out = self.linear1(x)  # (B, T, C) @ (C, C * 4) -> (B, T, C * 4)
        out = self.relu(out)  # (B, T, C * 4)
        out = self.linear2(out)  # (B, T, C * 4) @ (B, C * 4, C)
        return out  # (B, T, C)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ff = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        a = self.layernorm1(x)
        a = self.multihead_attn(a)
        a = x + a

        b = self.layernorm2(a)
        b = self.ff(b)
        b = a + b
        return b
