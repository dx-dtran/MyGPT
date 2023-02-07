import torch
import torch.nn as nn
import torch.nn.functional as F

# quick hyper params
# transformer
batch_size = 16
block_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# training
max_iters = 5000
eval_interval = 100
eval_iters = 200
learning_rate = 1e-3
device = 'cpu'

# transformer
# batch_size = 16
# block_size = 64
# n_embd = 128
# n_head = 8
# n_layer = 8
# dropout = 0.0
#
# # training
# max_iters = 5000
# eval_interval = 100
# eval_iters = 200
# learning_rate = 1e-3
# device = 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r') as input_file:
    input_data = input_file.read()

unique_chars = set(input_data)
unique_chars = sorted(list(unique_chars))
vocab_size = len(unique_chars)

atoi = {char: i for i, char in enumerate(unique_chars)}
itoa = {i: char for i, char in enumerate(unique_chars)}


def encode(data):
    return [atoi[char] for char in data]


def decode(indices):
    return ''.join([itoa[i] for i in indices])


data_tensor = torch.tensor(encode(input_data))

n = int(len(data_tensor) * 0.9)

train_data = data_tensor[:n]
val_data = data_tensor[n:]


def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    x = []
    y = []
    for i in range(batch_size):
        index = torch.randint(0, len(data) - block_size - 1, (1,))
        x.append(data[index:index + block_size])
        y.append(data[index + 1:index + block_size + 1])
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


def estimate_loss():
    model.eval()
    result = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[iteration] = loss
        result[split] = losses.mean()
    model.train()
    return result


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.query_matrix = nn.Linear(n_embd, head_size, bias=False)
        self.value_matrix = nn.Linear(n_embd, head_size, bias=False)

        # block_size is the context length (time dimension) that we want to mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        b, t, c = x.shape

        # order of matrix multiplication: the input tensor multiplied by the linear layer
        queries = self.query_matrix(x)  # (b, t, c) @ (c, h) -> (b, t, h)
        keys = self.key_matrix(x)  # (b, t, c) @ (c, h) -> (b, t, h)

        keys = keys.transpose(2, 1)  # (b, h, t)

        # order: the input tensor multiplied by the linear layer
        out = queries.bmm(keys)  # (b, t, h) @ (b, h, t) -> (b, t, t)
        out = out / (self.head_size ** 0.5)

        # what's another way to initialize tril here in the forward pass without requiring grad?
        # kind of weird to initialize it in the init method with a block_size shape, when we only need it to be
        # time_dim shape
        out = torch.masked_fill(out, self.tril[:t, :t] == 0.0, float('-inf'))  # (b, t, t)
        out = F.softmax(out, dim=2)

        values = self.value_matrix(x)  # (b, t, c) @ (c, h) -> (b, t, h)

        return out.bmm(values)  # (b, t, t) @ (b, t, h) -> (b, t, h)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = [head(x) for head in self.heads]  # (b, t, h)
        out = torch.cat(out, dim=2)  # (b, t, h * num_heads)
        return self.linear(out)  # (b, t, h * num_heads) @ (h * num_heads, c) -> (b, t, c)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.layer1 = nn.Linear(n_embd, n_embd * 4)
        self.layer2 = nn.Linear(n_embd * 4, n_embd)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)  # (b, t, c) @ (b, c, c * 4) -> (b, t, c * 4)
        out = self.relu(out)  # (b, t, c * 4)
        out = self.layer2(out)  # (b, t, c * 4) @ (b, c * 4, c) -> (b, t, c)
        return out  # (b, t, c)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ff = FeedForward(n_embd)
        self.norm = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn = self.norm(x)  # (b, t, c)
        attn = self.attn(attn)  # (b, t, c)
        attn = x + attn  # (b, t, c)

        ff = self.norm2(attn)  # (b, t, c)
        ff = self.ff(ff)  # (b, t, c)
        ff = attn + ff  # (b, t, c)
        return ff  # (b, t, c)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.positional_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.layernorm = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.token_embeddings(idx)  # (b, t, c)
        pos_emb = self.positional_embeddings(torch.arange(0, t))  # (t, c)
        emb = tok_emb + pos_emb  # (b, t, c)
        for block in self.blocks:
            emb = block(emb)  # (b, t, c)
        norm = self.layernorm(emb)  # (b, t, c)
        logits = self.linear(norm)  # (b, t, c) @ (c, v) -> (b, t, v)
        _, _, v = logits.shape
        if targets is not None:
            # targets original shape = (b, t)
            logits = logits.view(b * t, v)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        logits = logits.view(b * t, v)
        return logits, None

    def generate(self, idx, max_new_tokens):
        b, t = idx.shape
        result = torch.zeros(1, max_new_tokens)
        for i in range(max_new_tokens):
            idx = idx[:, len(idx) - block_size:]  # (b, t)
            logits, _ = self(idx)  # (b * t, v)
            probs = F.softmax(logits, dim=1)  # (b * t, v)
            index = torch.multinomial(probs[-1], 1)  # (b * t, v)
            index = index.view(b, t)  # (b, t)
            idx = torch.cat((idx, index), dim=1)
            result[:, i] = index
        return result


