import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv):
        super().__init__()
        self.d_qkv = d_qkv
        self.key_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.query_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.value_matrix = nn.Linear(d_embed, d_qkv, bias=False)

        # context_length is the context length (time dimension) that we want to mask
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        _, d_time, _ = x.shape

        # order of matrix multiplication: the input tensor multiplied by the linear layer
        queries = self.query_matrix(x)  # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)
        keys = self.key_matrix(x)  # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)

        keys = keys.transpose(2, 1)  # (d_batch, h, d_time)

        # order: the queries tensor multiplied by the keys tensor
        attention_matrix = queries.bmm(keys)  # (d_batch, d_time, h) @ (d_batch, h, d_time) -> (d_batch, d_time, d_time)
        attention_matrix = attention_matrix / (self.d_qkv ** 0.5)

        # what's another way to initialize tril here in the forward pass without requiring grad?
        # kind of weird to initialize it in the init method with a context_length shape, when we only need it to be
        # time_dim shape
        attention_matrix = torch.masked_fill(attention_matrix, self.mask[:d_time, :d_time] == 0.0, float('-inf'))  # (d_batch, d_time, d_time)
        attention_matrix = F.softmax(attention_matrix, dim=2)

        values = self.value_matrix(x)  # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)

        return attention_matrix.bmm(values)  # (d_batch, d_time, d_time) @ (d_batch, d_time, h) -> (d_batch, d_time, h)


class MultiSelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv, num_heads):
        super().__init__()
        self.self_attentions = nn.ModuleList([SelfAttention(context_length, d_embed, d_qkv) for _ in range(num_heads)])
        self.linear_proj = nn.Linear(d_qkv * num_heads, d_embed)

    def forward(self, x):
        out = [sa(x) for sa in self.self_attentions]  # (b, t, h)
        out = torch.cat(out, dim=2)  # (b, t, h * num_heads)
        return self.linear_proj(out)  # (b, t, h * num_heads) @ (h * num_heads, c) -> (b, t, c)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_proj1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_proj2 = nn.Linear(d_embed * 4, d_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear_proj1(x)  # (b, t, c) @ (b, c, c * 4) -> (b, t, c * 4)
        out = self.relu(out)  # (b, t, c * 4)
        out = self.linear_proj2(out)  # (b, t, c * 4) @ (b, c * 4, c) -> (b, t, c)
        return out  # (b, t, c)


class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_embed, n_head):
        super().__init__()
        self.attention = MultiSelfAttention(context_length, d_embed, d_embed // n_head, n_head)
        self.mlp = MultiLayerPerceptron(d_embed)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        attention = self.layer_norm1(x)  # (b, t, c)
        attention = self.attention(attention)  # (b, t, c)
        attention = x + attention  # (b, t, c)

        mlp = self.layer_norm2(attention)  # (b, t, c)
        mlp = self.mlp(mlp)  # (b, t, c)
        mlp = attention + mlp  # (b, t, c)
        return mlp  # (b, t, c)


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length=32, d_embed=64, n_head=4, n_layer=4):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_embed)
        self.positional_embeddings = nn.Embedding(context_length, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(context_length, d_embed, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, indices, targets=None):
        d_batch, d_time = indices.shape
        tok_emb = self.token_embeddings(indices)  # (d_batch, d_time, c)
        pos_emb = self.positional_embeddings(torch.arange(0, d_time))  # (d_time, c)
        emb = tok_emb + pos_emb  # (d_batch, d_time, c)
        for block in self.blocks:
            emb = block(emb)  # (d_batch, d_time, c)
        norm = self.layer_norm(emb)  # (d_batch, d_time, c)
        logits = self.linear(norm)  # (d_batch, d_time, c) @ (c, v) -> (d_batch, d_time, v)
        _, _, v = logits.shape
        if targets is not None:
            # targets original shape = (d_batch, d_time)
            logits = logits.view(d_batch * d_time, v)
            targets = targets.view(d_batch * d_time)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        logits = logits.view(d_batch * d_time, v)
        return logits, None

    def generate(self, idx, max_new_tokens=100):
        d_batch, d_time = idx.shape
        result = torch.zeros(1, max_new_tokens)
        for i in range(max_new_tokens):
            idx = idx[:, len(idx) - self.context_length:]  # (d_batch, d_time)
            logits, _ = self(idx)  # (d_batch * d_time, v)
            probs = F.softmax(logits, dim=1)  # (d_batch * d_time, v)
            index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, v)
            index = index.view(d_batch, d_time)  # (d_batch, d_time)
            idx = torch.cat((idx, index), dim=1)
            result[:, i] = index
        return result
