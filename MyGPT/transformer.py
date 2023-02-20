import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Transformer is a math function that transforms an input sequence of tokens into predictions of the next token
    It uses an attention mechanism to "pay attention" to the most important tokens
    By focusing on the most important tokens, it is easier to choose a new relevant token
    """

    def __init__(
            self, vocab_size, device, context_length=64, d_embed=128, n_head=8, n_layer=4
    ):
        super().__init__()
        self.device = device
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_embed)
        self.positional_embeddings = nn.Embedding(context_length, d_embed)
        self.transformer_blocks = nn.ModuleList(
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

        # repeatedly transform and simplify the representation of the tokens via attention and multilayer perceptrons
        for transformer_block in self.transformer_blocks:
            embedding = transformer_block(embedding)
        normalized = self.layer_norm(embedding)

        # project the transformed representation down to a set of scores for each token in the vocabulary
        logits = self.linear(normalized)
        _, _, vocab_size = logits.shape

        if targets is not None:
            logits = logits.view(d_batch * d_time, vocab_size)
            targets = targets.view(d_batch * d_time)

            # calculate the scores' correctness by measuring the difference between the scores and the known target
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        logits = logits.view(d_batch * d_time, vocab_size)
        return logits, None


class SelfAttention(nn.Module):
    """
    SelfAttention is how Transformer identifies (pays attention to) the most important tokens in a sequence
    For each token in the input, SelfAttention finds other tokens it has a strong relationship with
    It then extracts these tokens -- "pays attention" to them -- which simplifies the input
    This simplification process makes it easier to reason about the next token
    """

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
        # find the other tokens in the sequence that it has the strongest relationships with
        attention_matrix = F.softmax(attention_matrix, dim=2)

        # create another linear projection of the input sequence of tokens
        values = self.value_matrix(x)

        # matrix multiply the attention matrix (relationship strengths) with the values
        # this effectively focuses on the most important tokens in the input sequence
        # this simplification of the input makes it easier to make better predictions
        return attention_matrix.bmm(values)


class MultiSelfAttention(nn.Module):
    """
    MultiSelfAttention runs the attention mechanism multiple times on an input sequence
    It identifies complex relationships that may have been missed if attention had only been performed once
    """

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
        # repeat the self attention mechanism multiple times to find more relationships between tokens
        # this is done because self attention tends to find only one set of token relationships in the sequence
        out = [
            self_attention(x) for self_attention in self.self_attentions
        ]
        out = torch.cat(out, dim=2)
        return self.linear_proj(out)


class MultiLayerPerceptron(nn.Module):
    """
    A MultiLayerPerceptron transforms an input into a simplified representation
    It takes the features that have been extracted by attention and further simplifies them
    """

    def __init__(self, d_embed):
        super().__init__()
        self.linear_proj1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_proj2 = nn.Linear(d_embed * 4, d_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        # transform the input via two linear projections and a ReLU non-linear function in between
        hidden = self.linear_proj1(x)
        hidden = self.relu(hidden)
        out = self.linear_proj2(hidden)
        return out


class TransformerBlock(nn.Module):
    """
    TransformerBlock simplifies an input representation through two main mechanisms: attention and multilayer perceptron
    """

    def __init__(self, context_length, d_embed, n_head):
        super().__init__()
        self.attention = MultiSelfAttention(
            context_length, d_embed, d_embed // n_head, n_head
        )
        self.multilayer_perceptron = MultiLayerPerceptron(d_embed)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        # use the attention mechanism to obtain the most relevant token features of the input sequence
        attention = self.layer_norm1(x)
        attention = self.attention(attention)
        attention = x + attention

        # use a multilayer perceptron to further transform and simplify these features
        mlp = self.layer_norm2(attention)
        mlp = self.multilayer_perceptron(mlp)
        mlp = attention + mlp
        return mlp
