import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters

# ------------

torch.manual_seed(1337)


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


# here are all the unique characters that occur in this text

# create a mapping from characters to integers

# Train and test splits


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    pass


@torch.no_grad()
def estimate_loss():
    pass


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        pass

    def forward(self, x):
        # compute attention scores ("affinities")

        # perform the weighted aggregation of the values

        pass


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        pass

    def forward(self, x):
        pass


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()

    def forward(self, x):
        pass


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        pass

    def forward(self, x):
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        pass

    def forward(self, idx, targets=None):
        pass

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # crop idx to the last block_size tokens
        # get the predictions
        # focus only on the last time step
        # apply softmax to get probabilities
        # sample from the distribution
        # append sampled index to the running sequence
        pass

# print the number of parameters in the model

# create a PyTorch optimizer


# every once in a while evaluate the loss on train and val sets

# sample a batch of data

# evaluate the loss

# generate from the model
