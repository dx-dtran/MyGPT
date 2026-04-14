import json
import math
import torch
import torch.nn.functional as F
import os
import time

from MyGPT.transformer import Transformer
from MyGPT.generate import generate
from MyGPT.vocab import Tokenizer, create_vocabulary


def get_data(filename):
    try:
        with open(filename, "r") as input_file:
            input_data = input_file.read()
            return input_data
    except FileNotFoundError:
        print("data file not found")


def get_train_val_data(data, tokenizer, device, train_val_split=0.9):
    encoded_data = tokenizer.encode(data)
    data_tensor = torch.tensor(encoded_data, device=device).unsqueeze(0)
    n = int(data_tensor.shape[1] * train_val_split)
    train_data = data_tensor[0][:n]
    val_data = data_tensor[0][n:]
    return train_data, val_data


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


def pretrain(data_filename):
    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # obtain the dataset
    data_path = os.path.join("data", data_filename)
    raw_data = get_data(data_path)

    # create the vocabulary
    vocab, vocab_size = create_vocabulary(raw_data)
    tokenizer = Tokenizer(vocab)

    # convert the raw data to tensors
    train_data, val_data = get_train_val_data(raw_data, tokenizer, device)

    # define the training hyperparameters
    batch_size = 16
    max_iters = 5000
    eval_interval = 100
    eval_iters = 100
    learning_rate = 1e-3
    context_length = 64

    # define the model
    mygpt = Transformer(
        vocab_size,
        device,
        context_length=context_length,
        d_embed=128,
        n_head=8,
        n_layer=4,
    )
    mygpt.to(device)

    num_params = sum(param.numel() for param in mygpt.parameters())
    print("MyGPT initialized with {} parameters".format(num_params))
    print("Begin training using {}".format(device))

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    start = time.time()
    for iteration in range(max_iters):
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = estimate_loss(
                mygpt, train_data, batch_size, context_length, eval_iters
            )
            val_loss = estimate_loss(
                mygpt, val_data, batch_size, context_length, eval_iters
            )

            print("\n===========================================================================================")
            print(
                "iteration: {} | training loss: {:0.3f} | validation loss: {:0.3f} | elapsed: {:0.2f} seconds ".format(
                    iteration, train_loss, val_loss, time.time() - start
                )
            )
            print("===========================================================================================\n")

            context = torch.tensor([[0]], dtype=torch.long, device=device)
            generate(mygpt, context, tokenizer, num_new_tokens=200)

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = mygpt(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Total training time: {:0.2f} seconds".format(time.time() - start))

    # save the model weights

    # weights_path = os.path.join("weights", data_filename + ".pth")
    # torch.save(mygpt.state_dict(), weights_path)


# ── BPE sample generation ─────────────────────────────────────────────────────

def load_bpe_decoder(tokenizer_file="tokenizer.json"):
    with open(tokenizer_file, "r") as f:
        data = json.load(f)
    id_to_tok = {v: k for k, v in data["vocab"].items()}
    end_id = data["vocab"]["[END]"]
    return id_to_tok, end_id


@torch.no_grad()
def generate_sample(model, prompt_ids, id_to_tok, end_id, device, max_new=20):
    model.eval()
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_toks = []
    for _ in range(max_new):
        ctx = context[:, -model.context_length:]
        scores, _ = model(ctx)
        probs = torch.softmax(scores[-1], dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        output_toks.append(next_id)
        context = torch.cat([context, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == end_id:
            break
    model.train()
    prompt_str = "".join(id_to_tok.get(i, "?") for i in prompt_ids)
    response_str = "".join(id_to_tok.get(i, "?") for i in output_toks)
    print(f"  SAMPLE | {prompt_str}{response_str}")


# ── JSONL dataset helpers ─────────────────────────────────────────────────────

def load_jsonl(path, context_length):
    tokens_list, masks_list = [], []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            tokens_list.append(obj["tokens"][:context_length])
            masks_list.append(obj["loss_mask"][:context_length])
    tokens = torch.tensor(tokens_list, dtype=torch.long)
    masks = torch.tensor(masks_list, dtype=torch.float)
    return tokens, masks


def get_jsonl_batch(tokens, masks, indices, device):
    t = tokens[indices].to(device)
    m = masks[indices].to(device)
    x = t[:, :-1]                 # input
    y = t[:, 1:]                  # targets (shifted by 1)
    mask = m[:, 1:]               # 1 = compute loss (response tokens only)
    return x, y, mask


def masked_cross_entropy(scores, y, mask):
    loss = F.cross_entropy(scores, y.reshape(-1), reduction="none")
    loss = loss * mask.reshape(-1)
    return loss.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def estimate_jsonl_loss(model, tokens, masks, indices, device, eval_iters, batch_size):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        idx = indices[torch.randint(len(indices), (batch_size,))]
        x, y, mask = get_jsonl_batch(tokens, masks, idx, device)
        scores, _ = model(x)
        losses.append(masked_cross_entropy(scores, y, mask).item())
    model.train()
    return sum(losses) / len(losses)


# ── train_bike ────────────────────────────────────────────────────────────────

def train_bike(dataset_file="dataset_encoded.jsonl"):
    # Chinchilla-ish sizing for ~100k response tokens:
    # ~200k-param model trained 20 epochs ≈ 20x compute-optimal token budget.
    context_length = 32   # real sequences are ~21 tokens; 128 was mostly padding
    vocab_size = 761
    d_embed = 64
    n_head = 4
    n_layer = 2

    batch_size = 64
    epochs = 20
    learning_rate = 3e-3
    lr_min = 1e-4
    eval_interval = 200
    eval_iters = 50
    val_split = 0.1

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {dataset_file}...")
    tokens, masks = load_jsonl(dataset_file, context_length)
    N = len(tokens)
    n_val = max(1, int(N * val_split))
    perm = torch.randperm(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    # model sees sequences of length context_length-1 (x = tokens[:, :-1])
    mygpt = Transformer(
        vocab_size=vocab_size,
        device=device,
        context_length=context_length - 1,
        d_embed=d_embed,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)

    n_params = sum(p.numel() for p in mygpt.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Using device: {device}")

    id_to_tok, end_id = load_bpe_decoder()

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    steps_per_epoch = math.ceil(len(train_idx) / batch_size)
    total_steps = epochs * steps_per_epoch

    os.makedirs("weights", exist_ok=True)
    weights_path = os.path.join("weights", "bike.pth")

    start = time.time()
    step = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_perm = train_idx[torch.randperm(len(train_idx))]
        for batch_start in range(0, len(epoch_perm), batch_size):
            idx = epoch_perm[batch_start: batch_start + batch_size]

            # cosine LR decay
            progress = step / max(total_steps - 1, 1)
            lr = lr_min + 0.5 * (learning_rate - lr_min) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y, mask = get_jsonl_batch(tokens, masks, idx, device)
            scores, _ = mygpt(x)
            loss = masked_cross_entropy(scores, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_interval == 0 or step == total_steps - 1:
                val_loss = estimate_jsonl_loss(
                    mygpt, tokens, masks, val_idx, device, eval_iters, batch_size
                )
                print(
                    "step {:5d} | epoch {:2d} | lr {:.5f} | "
                    "train loss {:.4f} | val loss {:.4f} | {:.1f}s".format(
                        step, epoch, lr, loss.item(), val_loss, time.time() - start
                    )
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(mygpt.state_dict(), weights_path)

                # pick a random val example and generate from its prompt
                sample_idx = val_idx[torch.randint(len(val_idx), (1,)).item()]
                sample_toks = tokens[sample_idx].tolist()
                # prompt = everything up to and including [A] (where mask first turns on)
                sample_mask = masks[sample_idx].tolist()
                prompt_end = next(
                    (i for i, m in enumerate(sample_mask) if m == 1.0), len(sample_toks)
                )
                prompt_ids = sample_toks[:prompt_end]
                generate_sample(mygpt, prompt_ids, id_to_tok, end_id, device)

            step += 1

    print("\nTotal training time: {:.2f} seconds".format(time.time() - start))
    print("Best val loss: {:.4f}".format(best_val_loss))
    print("Weights saved to {}".format(weights_path))
