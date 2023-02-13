import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv, device):
        super().__init__()
        self.d_qkv = d_qkv
        self.query_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.key_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.value_matrix = nn.Linear(d_embed, d_qkv, bias=False)

        # context_length is the context length (time dimension) that we want to mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length, device=device)),
        )

    def forward(self, x):
        d_batch, d_time, d_emb = x.shape

        # order of matrix multiplication: the input tensor multiplied by the linear layer
        # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)
        queries = self.query_matrix(x)

        # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)
        keys = self.key_matrix(x)

        keys = keys.transpose(2, 1)  # (d_batch, h, d_time)

        # order: the queries tensor multiplied by the keys tensor
        # (d_batch, d_time, h) @ (d_batch, h, d_time) -> (d_batch, d_time, d_time)
        attention_matrix = queries.bmm(keys)
        attention_matrix = attention_matrix / (self.d_qkv ** 0.5)

        # what's another way to initialize tril here in the forward pass without requiring grad?
        # kind of weird to initialize it in the init method with a context_length shape, when we only need it to be
        # d_time shape
        attention_matrix = torch.masked_fill(
            attention_matrix, self.mask[:d_time, :d_time] == 0.0, float("-inf")
        )  # (d_batch, d_time, d_time)
        attention_matrix = F.softmax(attention_matrix, dim=2)

        # (d_batch, d_time, d_emb) @ (d_emb, h) -> (d_batch, d_time, h)
        values = self.value_matrix(x)

        # (d_batch, d_time, d_time) @ (d_batch, d_time, h) -> (d_batch, d_time, h)
        return attention_matrix.bmm(values)


class MultiSelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv, num_heads, device):
        super().__init__()
        self.self_attentions = nn.ModuleList(
            [
                SelfAttention(context_length, d_embed, d_qkv, device)
                for _ in range(num_heads)
            ]
        )
        self.linear_proj = nn.Linear(d_qkv * num_heads, d_embed)

    def forward(self, x):
        out = [
            self_attention(x) for self_attention in self.self_attentions
        ]  # (d_batch, d_time, h)
        out = torch.cat(out, dim=2)  # (d_batch, d_time, h * num_heads)
        # (d_batch, d_time, h * num_heads) @ (h * num_heads, d_embed) -> (d_batch, d_time, d_embed)
        return self.linear_proj(out)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_proj1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_proj2 = nn.Linear(d_embed * 4, d_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (d_batch, d_time, d_embed) @ (d_batch, d_embed, d_embed * 4) -> (d_batch, d_time, d_embed * 4)
        hidden = self.linear_proj1(x)

        # (d_batch, d_time, d_embed * 4)
        hidden = self.relu(hidden)

        # (d_batch, d_time, d_embed * 4) @ (d_batch, d_embed * 4, d_embed) -> (d_batch, d_time, d_embed)
        out = self.linear_proj2(hidden)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_embed, n_head, device):
        super().__init__()
        self.attention = MultiSelfAttention(
            context_length, d_embed, d_embed // n_head, n_head, device
        )
        self.mlp = MultiLayerPerceptron(d_embed)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        attention = self.layer_norm1(x)  # (d_batch, d_time, d_embed)
        attention = self.attention(attention)  # (d_batch, d_time, d_embed)
        attention = x + attention  # (d_batch, d_time, d_embed)

        mlp = self.layer_norm2(attention)  # (d_batch, d_time, d_embed)
        mlp = self.mlp(mlp)  # (d_batch, d_time, d_embed)
        mlp = attention + mlp  # (d_batch, d_time, d_embed)
        return mlp  # (d_batch, d_time, d_embed)


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, device, context_length=64, d_embed=128, n_head=8, n_layer=4
    ):
        super().__init__()
        self.device = device
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_embed)
        self.positional_embeddings = nn.Embedding(context_length, d_embed)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(context_length, d_embed, n_head, device)
                for _ in range(n_layer)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, indices, targets=None):
        d_batch, d_time = indices.shape
        token_embedding = self.token_embeddings(indices)  # (d_batch, d_time, d_embed)
        positional_embedding = self.positional_embeddings(
            torch.arange(0, d_time, device=self.device)
        )  # (d_time, d_embed)
        embedding = token_embedding + positional_embedding  # (d_batch, d_time, d_embed)
        for block in self.blocks:
            embedding = block(embedding)  # (d_batch, d_time, d_embed)
        normalized = self.layer_norm(embedding)  # (d_batch, d_time, d_embed)

        # (d_batch, d_time, d_embed) @ (d_embed, vocab_size) -> (d_batch, d_time, vocab_size)
        logits = self.linear(normalized)
        _, _, vocab_size = logits.shape

        if targets is not None:
            # targets original shape = (d_batch, d_time)
            logits = logits.view(d_batch * d_time, vocab_size)
            targets = targets.view(d_batch * d_time)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        logits = logits.view(d_batch * d_time, vocab_size)
        return logits, None
