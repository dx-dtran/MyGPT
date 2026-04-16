import json
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a")

    def log(self, msg=""):
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def create_vocabulary(data):
    vocab = sorted(set(data))
    return vocab, len(vocab)


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._atoi = {ch: i for i, ch in enumerate(vocab)}
        self._itoa = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, text):
        return [self._atoi[ch] for ch in text]

    def decode(self, indices):
        return "".join(self._itoa[i] for i in indices)


class SelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv):
        super().__init__()
        self.d_qkv = d_qkv
        self.query_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.key_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.value_matrix = nn.Linear(d_embed, d_qkv, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        d_batch, d_time, _ = x.shape
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        attention_matrix = queries.bmm(keys.transpose(2, 1)) / (self.d_qkv ** 0.5)
        attention_matrix = torch.masked_fill(attention_matrix, self.mask[:d_time, :d_time] == 0.0, float("-inf"))
        attention_matrix = F.softmax(attention_matrix, dim=2)
        values = self.value_matrix(x)
        return attention_matrix.bmm(values)


class MultiSelfAttention(nn.Module):
    def __init__(self, context_length, d_embed, d_qkv, n_head):
        super().__init__()
        self.self_attentions = nn.ModuleList(
            [SelfAttention(context_length, d_embed, d_qkv) for _ in range(n_head)]
        )
        self.linear_proj = nn.Linear(d_qkv * n_head, d_embed)

    def forward(self, x):
        out = [sa(x) for sa in self.self_attentions]
        return self.linear_proj(torch.cat(out, dim=2))


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_proj1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_proj2 = nn.Linear(d_embed * 4, d_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.linear_proj1(x)
        hidden = self.relu(hidden)
        return self.linear_proj2(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_embed, n_head):
        super().__init__()
        self.attention = MultiSelfAttention(context_length, d_embed, d_embed // n_head, n_head)
        self.multilayer_perceptron = MultiLayerPerceptron(d_embed)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        attention = self.layer_norm1(x)
        attention = self.attention(attention)
        attention = x + attention
        mlp = self.layer_norm2(attention)
        mlp = self.multilayer_perceptron(mlp)
        return attention + mlp


class Transformer(nn.Module):
    def __init__(self, vocab_size, device, context_length=64, d_embed=64, n_head=4, n_layer=2):
        super().__init__()
        self.device = device
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_embed)
        self.positional_embeddings = nn.Embedding(context_length, d_embed)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(context_length, d_embed, n_head) for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, indices, targets=None):
        d_batch, d_time = indices.shape
        token_embedding = self.token_embeddings(indices)
        positional_embedding = self.positional_embeddings(torch.arange(0, d_time, device=self.device))
        embedding = token_embedding + positional_embedding
        for transformer_block in self.transformer_blocks:
            embedding = transformer_block(embedding)
        normalized = self.layer_norm(embedding)
        scores = self.linear(normalized)
        _, _, vocab_size = scores.shape
        if targets is not None:
            scores = scores.view(d_batch * d_time, vocab_size)
            targets = targets.view(d_batch * d_time)
            loss = F.cross_entropy(scores, targets)
            return scores, loss
        scores = scores.view(d_batch * d_time, vocab_size)
        return scores, None


def get_data(path):
    with open(path, "r") as f:
        return f.read()


def get_train_val_data(data, tokenizer, device, train_val_split=0.9):
    encoded_data = tokenizer.encode(data)
    data_tensor = torch.tensor(encoded_data, device=device)
    n = int(len(data_tensor) * train_val_split)
    return data_tensor[:n], data_tensor[n:]


def get_batch(data, batch_size, context_length):
    x, y = [], []
    for i in range(batch_size):
        index = torch.randint(0, len(data) - context_length - 1, (1,))
        x.append(data[index: index + context_length])
        y.append(data[index + 1: index + context_length + 1])
    x, y = torch.stack(x), torch.stack(y)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for iteration in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length)
        _, loss = model(x, y)
        losses[iteration] = loss
    model.train()
    return losses.mean()


def _fmt_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    return f"{flops / 1e9:.1f} GFLOPs"


def _fmt_throughput(flops_per_s):
    g = flops_per_s / 1e9
    if g >= 1.0:
        return f"{g:.2f} GFLOPs/s"
    return f"{g * 1000:.0f} MFLOPs/s"


def _fmt_eta(seconds):
    s = int(seconds)
    if s <= 0:
        return "done"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def train():
    context_length = 64
    d_embed = 64
    n_head = 4
    n_layer = 2

    batch_size = 64
    max_iters = 2000
    eval_interval = 500
    eval_iters = 10
    learning_rate = 3e-3
    lr_min = 1e-4
    train_val_split = 0.9

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = os.path.join(_DIR, "dataset.txt")
    weights_path = os.path.join(_DIR, "weights", "bike_char.pth")
    vocab_path = os.path.join(_DIR, "weights", "bike_char_vocab.json")
    log_path = os.path.join(_DIR, "logs", "bike_char.log")

    os.makedirs(os.path.join(_DIR, "weights"), exist_ok=True)

    log = Logger(log_path)
    log.log("=== train_bike_char  {} ===".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    log.log(f"loading {data_path}...")

    raw_data = get_data(data_path)
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    train_data, val_data = get_train_val_data(raw_data, tokenizer, device, train_val_split)

    mygpt = Transformer(
        vocab_size=vocab_size,
        device=device,
        context_length=context_length,
        d_embed=d_embed,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)

    n_params = sum(p.numel() for p in mygpt.parameters())
    flops_per_step = 6 * n_params * batch_size * context_length

    log.log(f"vocab size    : {vocab_size} chars")
    log.log(f"train tokens  : {len(train_data):,}   val tokens: {len(val_data):,}")
    log.log(f"parameters    : {n_params:,}")
    log.log(f"context length: {context_length}   d_embed: {d_embed}   layers: {n_layer}   heads: {n_head}")
    log.log(f"batch size    : {batch_size}   max iters: {max_iters}   lr: {learning_rate} -> {lr_min}")
    log.log(f"device        : {device}")
    log.log(f"flops/step    : {_fmt_flops(flops_per_step)}")
    log.log("")

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    start = time.time()
    best_val_loss = float("inf")
    cumulative_flops = 0
    window_start = start
    window_flops = 0

    for step in range(max_iters):
        progress = step / max(max_iters - 1, 1)
        lr = lr_min + 0.5 * (learning_rate - lr_min) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = mygpt(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumulative_flops += flops_per_step
        window_flops += flops_per_step

        if step % eval_interval == 0 or step == max_iters - 1:
            eval_start = time.time()
            window_elapsed = max(eval_start - window_start, 1e-6)
            flops_per_s = window_flops / window_elapsed

            train_loss = estimate_loss(mygpt, train_data, batch_size, context_length, eval_iters)
            val_loss = estimate_loss(mygpt, val_data, batch_size, context_length, eval_iters)
            saved = val_loss < best_val_loss
            if saved:
                best_val_loss = val_loss
                torch.save(mygpt.state_dict(), weights_path)

            sample_chars = []
            with torch.no_grad():
                mygpt.eval()
                sample_ctx = torch.tensor([[0]], dtype=torch.long, device=device)
                for _ in range(120):
                    c = sample_ctx[:, -mygpt.context_length:]
                    scores, _ = mygpt(c)
                    probs = torch.softmax(scores[-1], dim=-1)
                    nxt = torch.multinomial(probs, 1).item()
                    sample_chars.append(tokenizer.decode([nxt]))
                    sample_ctx = torch.cat([sample_ctx, torch.tensor([[nxt]], device=device)], dim=1)
                mygpt.train()

            elapsed = time.time() - start
            steps_done = step + 1
            eta_s = (max_iters - steps_done) * (elapsed / steps_done)

            tag = " [saved]" if saved else ""
            log.log("step {}/{} | {} | lr {:.5f} | train {:.4f} | val {:.4f} | {:.1f}s | {} | {} | eta {}{}".format(
                step, max_iters,
                time.strftime("%H:%M:%S"),
                lr, train_loss, val_loss, elapsed,
                _fmt_flops(cumulative_flops),
                _fmt_throughput(flops_per_s),
                _fmt_eta(eta_s),
                tag
            ))
            log.log("")
            log.log("sample text:")
            log.log("".join(sample_chars))
            log.log("")

            window_start = time.time()
            window_flops = 0

    log.log("total time : {:.1f}s".format(time.time() - start))
    log.log("best val   : {:.4f}".format(best_val_loss))
    log.log("weights    : {}".format(weights_path))
    log.close()


def chat(
    context_length=64,
    d_embed=64,
    n_head=4,
    n_layer=2,
    max_new=200,
    temperature=0.8,
):
    weights_path = os.path.join(_DIR, "weights", "bike_char.pth")
    vocab_path = os.path.join(_DIR, "weights", "bike_char_vocab.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    char_to_id = {ch: i for i, ch in enumerate(vocab)}

    def encode(text):
        return [char_to_id.get(ch, 0) for ch in text]

    def decode(ids):
        return "".join(id_to_char.get(i, "?") for i in ids)

    model = Transformer(
        vocab_size=vocab_size,
        device=device,
        context_length=context_length,
        d_embed=d_embed,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print("bike model loaded. type your question or 'quit' to exit.\n")

    with torch.no_grad():
        while True:
            user_input = input("you: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            user_input = user_input.lower()
            prompt = f"[H] {user_input} [A]"
            prompt_ids = encode(prompt)

            if len(prompt_ids) >= context_length:
                prompt_ids = prompt_ids[-(context_length - 1):]

            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_chars = []

            for _ in range(max_new):
                ctx = context[:, -context_length:]
                scores, _ = model(ctx)
                logits = scores[-1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                output_chars.append(id_to_char.get(next_id, "?"))
                context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)
                response_so_far = "".join(output_chars)
                if "[END]" in response_so_far:
                    output_chars = list(response_so_far.split("[END]")[0])
                    break

            print(f"bike: {''.join(output_chars).strip()}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        train()
