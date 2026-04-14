"""
Pure Python Corpus Encoder
Reads dataset.txt, encodes every line to fixed-length token IDs + loss mask.
No imports except json, os.

Requires:
    dataset.txt       - raw text dataset (one training example per line)
    tokenizer.json    - trained tokenizer (from train_tokenizer.py)

Usage:
    python encode_dataset.py

Output:
    dataset_encoded.jsonl   - one {"tokens": [...], "loss_mask": [...]} per line
    encode_stats.json       - stats about the encoded dataset
"""

import json
import os
import re

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE = "dataset.txt"
TOKENIZER_FILE = "tokenizer.json"
OUTPUT_FILE = "dataset_encoded.jsonl"
STATS_FILE = "encode_stats.json"
MAX_LEN = 128
LOG_EVERY = 100_000


# ── Load Tokenizer ────────────────────────────────────────────────────────────

def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data["special_tokens"],
        data["merges"],
        data["vocab"],
    )


# ── Pretokenizer (same as trainer) ───────────────────────────────────────────

def make_pretokenizer(special_tokens):
    pattern = "(" + "|".join(re.escape(s) for s in special_tokens) + r"|\S+)"
    return re.compile(pattern)


def pretokenize(text, pattern):
    return pattern.findall(text)


# ── BPE Encoder ───────────────────────────────────────────────────────────────

def encode(text, merges, vocab, special_set, pretok_pattern, unk_id):
    words = pretokenize(text, pretok_pattern)
    ids = []
    for word in words:
        if word in special_set:
            ids.append(vocab.get(word, unk_id))
        else:
            tokens = list(word)
            for pair in merges:
                merged = pair[0] + pair[1]
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if (i < len(tokens) - 1
                            and tokens[i] == pair[0]
                            and tokens[i + 1] == pair[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            for tok in tokens:
                ids.append(vocab.get(tok, unk_id))
    return ids


# ── Pad + Mask ────────────────────────────────────────────────────────────────

def pad_and_mask(ids, max_len, pad_id, a_id, end_id):
    # truncate if somehow over max_len
    ids = ids[:max_len]

    # build loss mask
    # 0 = ignore, 1 = compute loss
    # loss is on from the token AFTER [A] up to and including [END]
    loss_mask = [0] * len(ids)
    mask_on = False
    for i, tok in enumerate(ids):
        if tok == a_id:
            mask_on = True  # start masking after [A]
            continue  # [A] itself is not included in loss
        if mask_on:
            loss_mask[i] = 1
        if tok == end_id:
            mask_on = False  # stop after [END]

    # pad to max_len
    pad_len = max_len - len(ids)
    ids = ids + [pad_id] * pad_len
    loss_mask = loss_mask + [0] * pad_len

    return ids, loss_mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # check files exist
    for path in [INPUT_FILE, TOKENIZER_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

    # load tokenizer
    print(f"Loading tokenizer from {TOKENIZER_FILE}...")
    special_tokens, merges, vocab = load_tokenizer(TOKENIZER_FILE)

    special_set = set(special_tokens)
    pretok_pattern = make_pretokenizer(special_tokens)

    pad_id = vocab["[PAD]"]  # 0
    unk_id = vocab["[UNK]"]  # 1
    a_id = vocab["[A]"]  # 3
    end_id = vocab["[END]"]  # 4

    print(f"Vocab size : {len(vocab)}")
    print(f"[PAD] id   : {pad_id}")
    print(f"[UNK] id   : {unk_id}")
    print(f"[H]   id   : {vocab['[H]']}")
    print(f"[A]   id   : {a_id}")
    print(f"[END] id   : {end_id}")
    print(f"Max length : {MAX_LEN}")
    print()

    # encode
    print(f"Encoding {INPUT_FILE} -> {OUTPUT_FILE}...")

    n_lines = 0
    n_truncated = 0
    n_unk = 0
    n_empty_mask = 0
    total_tokens = 0
    total_response_tokens = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue

            # encode to ids
            ids = encode(line, merges, vocab, special_set, pretok_pattern, unk_id)

            # track stats
            if len(ids) > MAX_LEN:
                n_truncated += 1
            n_unk += ids.count(unk_id)

            # pad + build loss mask
            tokens, loss_mask = pad_and_mask(ids, MAX_LEN, pad_id, a_id, end_id)

            response_toks = sum(loss_mask)
            if response_toks == 0:
                n_empty_mask += 1  # line had no [A] token — something is wrong

            total_tokens += MAX_LEN
            total_response_tokens += response_toks
            n_lines += 1

            fout.write(
                json.dumps(
                    {
                        "tokens": tokens,
                        "loss_mask": loss_mask,
                    }
                ) + "\n"
                )

            if n_lines % LOG_EVERY == 0:
                print(f"  {n_lines:,} lines encoded...")

    # stats
    stats = {
        "total_lines": n_lines,
        "total_tokens": total_tokens,
        "total_response_tokens": total_response_tokens,
        "avg_response_tokens": round(total_response_tokens / max(n_lines, 1), 2),
        "truncated_lines": n_truncated,
        "unk_tokens": n_unk,
        "empty_mask_lines": n_empty_mask,
        "max_len": MAX_LEN,
        "output_file": OUTPUT_FILE,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # print summary
    print(
        f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Encoding complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Lines encoded         : {n_lines:,}
  Total tokens          : {total_tokens:,}
  Avg response tokens   : {stats['avg_response_tokens']}
  Truncated lines       : {n_truncated:,}
  UNK tokens            : {n_unk:,}
  Empty mask lines      : {n_empty_mask:,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        )

    if n_empty_mask > 0:
        print(
            f"WARNING: {n_empty_mask:,} lines had no [A] token and will contribute "
            f"zero loss during training. Check your dataset format."
            )
    if n_unk > 0:
        print(
            f"WARNING: {n_unk:,} UNK tokens found. Check your tokenizer covers "
            f"all characters in the dataset."
            )
    if n_truncated > 0:
        print(
            f"NOTE: {n_truncated:,} lines were truncated to {MAX_LEN} tokens. "
            f"Consider increasing MAX_LEN or shortening templates."
            )


if __name__ == "__main__":
    main()
