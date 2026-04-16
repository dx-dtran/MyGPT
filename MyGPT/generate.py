import json
import re
import torch
import torch.nn.functional as F
import os

from MyGPT.vocab import Tokenizer, get_vocabulary
from MyGPT.transformer import Transformer


def generate_next_token(model, context, tokenizer):
    d_batch, _ = context.shape
    scores, _ = model(context)  # (d_batch * d_time, vocab_size)
    probs = F.softmax(scores, dim=1)  # (d_batch * d_time, vocab_size)
    index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
    index = index.view(d_batch, 1)  # (d_batch, d_time)
    next_token = tokenizer.decode(index[0].tolist())
    return next_token, index


def generate(model, context, tokenizer, num_new_tokens=500, log_file=None):
    chars = []
    for _ in range(num_new_tokens):
        context = context[:, len(context) - model.context_length:]  # (d_batch, d_time)
        next_token, index = generate_next_token(model, context, tokenizer)
        chars.append(next_token)
        context = torch.cat((context, index), dim=1)
    text = "".join(chars)
    print(text)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()


def chat_bike(
    weights_path="weights/bike.pth",
    tokenizer_file="tokenizer.json",
    context_length=32,
    d_embed=64,
    n_head=4,
    n_layer=2,
    max_new=40,
    temperature=0.8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    with open(tokenizer_file, "r") as f:
        tok_data = json.load(f)
    vocab = tok_data["vocab"]
    merges = tok_data["merges"]
    special_tokens = tok_data["special_tokens"]
    special_set = set(special_tokens)
    id_to_tok = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    pad_id  = vocab["[PAD]"]
    unk_id  = vocab["[UNK]"]
    h_id    = vocab["[H]"]
    a_id    = vocab["[A]"]
    end_id  = vocab["[END]"]

    pretok_pattern = re.compile(
        "(" + "|".join(re.escape(s) for s in special_tokens) + r"|\S+)"
    )

    def bpe_encode(text):
        words = pretok_pattern.findall(text)
        ids = []
        for word in words:
            if word in special_set:
                ids.append(vocab.get(word, unk_id))
            else:
                tokens = list(word)
                for pair in merges:
                    merged = pair[0] + pair[1]
                    i, new_tokens = 0, []
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                            new_tokens.append(merged)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                for tok in tokens:
                    ids.append(vocab.get(tok, unk_id))
        return ids

    def bpe_decode(ids):
        special = {"[H]", "[A]", "[END]", "[PAD]", "[UNK]"}
        parts = []
        for i in ids:
            tok = id_to_tok.get(i, "?")
            parts.append(tok if tok in special else " " + tok)
        return "".join(parts).strip()

    # load model
    model = Transformer(
        vocab_size=vocab_size,
        device=device,
        context_length=context_length - 1,
        d_embed=d_embed,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"MyGPT bike model loaded. Type your question (or 'quit' to exit).\n")

    with torch.no_grad():
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            # lowercase to avoid UNK (dataset is lowercase)
            user_input = user_input.lower()

            # encode as [H] question [A]
            prompt_ids = bpe_encode(f"[H] {user_input} [A]")

            # warn if any UNK slipped through
            n_unk = prompt_ids.count(unk_id)
            if n_unk:
                print(f"  (warning: {n_unk} unknown token(s) in prompt)")

            # trim prompt to fit context, always keep [H] and [A]
            max_prompt = context_length - 2
            if len(prompt_ids) > max_prompt:
                prompt_ids = [h_id] + prompt_ids[-(max_prompt - 1):]

            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_ids = []

            for _ in range(max_new):
                ctx = context[:, -model.context_length:]
                scores, _ = model(ctx)
                logits = scores[-1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                output_ids.append(next_id)
                context = torch.cat(
                    [context, torch.tensor([[next_id]], device=device)], dim=1
                )
                if next_id == end_id:
                    break

            # strip trailing [END] for clean display
            if output_ids and output_ids[-1] == end_id:
                output_ids = output_ids[:-1]

            print(f"HungryGPT: {bpe_decode(output_ids)}\n")


def chat_bike_char(
    weights_path="weights/bike_char.pth",
    vocab_path="weights/bike_char_vocab.json",
    context_length=64,
    d_embed=96,
    n_head=4,
    n_layer=3,
    max_new=200,
    temperature=0.8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load char vocab (list of chars in index order)
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}

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

    print(f"MyGPT char-level bike model loaded. Type your question (or 'quit' to exit).\n")

    end_marker = "[END]"

    with torch.no_grad():
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            user_input = user_input.lower()
            prompt = f"[H] {user_input} [A]"
            prompt_ids = encode(prompt)

            # trim prompt to fit context window
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
                context = torch.cat(
                    [context, torch.tensor([[next_id]], device=device)], dim=1
                )
                response_so_far = "".join(output_chars)
                if end_marker in response_so_far:
                    response_so_far = response_so_far.split(end_marker)[0]
                    output_chars = list(response_so_far)
                    break

            print(f"HungryGPT: {''.join(output_chars).strip()}\n")


def generate_from_pretrained(data_filename, num_prompts=20, num_tokens=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = os.path.join("weights", "vocab.json")
    vocab, vocab_size = get_vocabulary(vocab_path, data_filename)
    tokenizer = Tokenizer(vocab)

    mygpt = Transformer(vocab_size, device)
    mygpt.to(device)

    weights_path = os.path.join("weights", data_filename + ".pth")
    mygpt.load_state_dict(torch.load(weights_path))

    for _ in range(num_prompts):
        prompt = tokenizer.encode(input("PROMPT: "))
        prompt = torch.tensor(prompt, device=device).unsqueeze(0)
        generate(mygpt, prompt, tokenizer, num_new_tokens=num_tokens)
