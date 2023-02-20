import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv):
        super().__init__()
        self.d_qkv = d_qkv
        self.query_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.key_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.value_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length)),
        )

    def forward(self, x):
        d_batch, d_time, d_emb = x.shape

        # create two separate linear projections of the same input sequence of tokens, and call them queries and keys
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)

        # matrix multiply the queries with the keys to create a (num_token by num_token) attention matrix
        # the attention matrix represents the strength of the relationships between each token in the sequence
        attention_matrix = queries.bmm(keys.transpose(2, 1))
        attention_matrix = attention_matrix / (self.d_qkv ** 0.5)

        # mask out future tokens in the sequence so that tokens in the past cannot reason about tokens in the future
        attention_matrix = torch.masked_fill(
            attention_matrix, self.mask[:d_time, :d_time] == 0.0, float("-inf")
        )

        # for each token in the sequence,
        # find the other token in the sequence that it has the strongest relationship with
        attention_matrix = F.softmax(attention_matrix, dim=2)

        # create another linear projection of the input sequence of tokens
        values = self.value_matrix(x)

        # matrix multiply the attention matrix (relationship strengths) with the values
        # this effectively focuses on the most important tokens in the input sequence
        # this simplification of the input sequence makes it easier to make better predictions
        return attention_matrix.bmm(values)


class MultiSelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv, num_heads):
        super().__init__()
        self.self_attentions = nn.ModuleList(
            [
                SelfAttention(context_length, d_embed, d_qkv)
                for _ in range(num_heads)
            ]
        )
        self.linear_proj = nn.Linear(d_qkv * num_heads, d_embed)

    def forward(self, x):
        # the softmax operation in self_attention makes it such that each token focuses on only one other token
        # repeat the self attention operation multiple times to learn the relationships between other tokens as well
        out = [
            self_attention(x) for self_attention in self.self_attentions
        ]
        out = torch.cat(out, dim=2)
        return self.linear_proj(out)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_proj1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_proj2 = nn.Linear(d_embed * 4, d_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        # transform the input into a simplified representation that makes it easier to make predictions
        hidden = self.linear_proj1(x)
        hidden = self.relu(hidden)
        out = self.linear_proj2(hidden)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_embed, n_head):
        super().__init__()
        self.attention = MultiSelfAttention(
            context_length, d_embed, d_embed // n_head, n_head
        )
        self.mlp = MultiLayerPerceptron(d_embed)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        # use the attention operations to obtain the most important features of the input sequence
        attention = self.layer_norm1(x)
        attention = self.attention(attention)
        attention = x + attention

        # use a multilayer perceptron to further transform and simplify these features
        mlp = self.layer_norm2(attention)
        mlp = self.mlp(mlp)
        mlp = attention + mlp
        return mlp


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
                TransformerBlock(context_length, d_embed, n_head)
                for _ in range(n_layer)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, indices, targets=None):
        d_batch, d_time = indices.shape

        # create an initial representation (called an "embedding") of the input tokens
        token_embedding = self.token_embeddings(indices)
        positional_embedding = self.positional_embeddings(
            torch.arange(0, d_time, device=self.device)
        )
        embedding = token_embedding + positional_embedding

        # repeatedly transform and simplify the representation of the tokens via attention and multilayer perceptron
        for block in self.blocks:
            embedding = block(embedding)
        normalized = self.layer_norm(embedding)

        # project the transformed representation down to a set of scores for each token in the vocabulary
        logits = self.linear(normalized)
        _, _, vocab_size = logits.shape

        if targets is not None:
            logits = logits.view(d_batch * d_time, vocab_size)
            targets = targets.view(d_batch * d_time)

            # calculate the correctness of the scores by measuring the difference between the scores and the target
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        logits = logits.view(d_batch * d_time, vocab_size)
        return logits, None
