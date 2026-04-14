"""
Pure Python BPE Tokenizer Trainer
No imports except json, collections, re, os.

Trains on a sample of dataset.txt and saves:
    tokenizer.json   - vocab + merge rules

Usage:
    python train_tokenizer.py
"""

import json
import re
import os
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE = "dataset.txt"
OUTPUT_FILE = "tokenizer.json"
VOCAB_SIZE = 1024
TRAIN_LINES = 10_000  # sample size to train on
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[H]", "[A]", "[END]"]

# ── Pretokenizer ──────────────────────────────────────────────────────────────
# splits text into words, keeping special tokens atomic

# build a regex that splits on special tokens first, then whitespace
# special tokens are matched as whole units and never split further
_SPECIAL_PAT = "(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + r"|\S+)"


def pretokenize(text):
    """Split text into words. Special tokens come out as atomic units."""
    return re.findall(_SPECIAL_PAT, text)


# ── BPE Core ──────────────────────────────────────────────────────────────────

def word_to_chars(word):
    """Turn a word into a tuple of characters (the initial BPE representation)."""
    return tuple(word)


def get_pair_counts(vocab):
    """
    Count all adjacent pairs across the whole vocab.
    vocab: dict of {tuple_of_tokens: frequency}
    returns: Counter of {(tok_a, tok_b): count}
    """
    counts = Counter()
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            counts[(tokens[i], tokens[i + 1])] += freq
    return counts


def merge_pair(vocab, pair):
    """
    Merge all occurrences of `pair` in every entry of vocab.
    Returns a new vocab dict.
    """
    merged = "".join(pair)
    new_vocab = {}
    for tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq
    return new_vocab


# ── Trainer ───────────────────────────────────────────────────────────────────

def train(lines, vocab_size, special_tokens):
    print(f"Pretokenizing {len(lines):,} lines...")

    # count word frequencies, skipping special tokens
    # (special tokens are pre-assigned IDs and never participate in BPE)
    special_set = set(special_tokens)
    word_counts = Counter()
    for line in lines:
        words = pretokenize(line)
        for word in words:
            if word not in special_set:
                word_counts[word] += 1

    print(f"Unique words in sample: {len(word_counts):,}")

    # initialize vocab: each word is a tuple of characters
    bpe_vocab = {word_to_chars(word): freq for word, freq in word_counts.items()}

    # collect all unique characters as the base vocab
    base_chars = set()
    for tokens in bpe_vocab:
        for ch in tokens:
            base_chars.add(ch)

    print(f"Base characters: {len(base_chars)}")

    # how many merges can we do?
    # vocab_size = special tokens + base chars + merges
    n_merges = vocab_size - len(special_tokens) - len(base_chars)
    print(f"Training {n_merges} BPE merges...")

    if n_merges <= 0:
        print("WARNING: vocab_size too small for all base chars + special tokens.")
        n_merges = 0

    merges = []  # ordered list of (pair_a, pair_b) merge rules

    for i in range(n_merges):
        pair_counts = get_pair_counts(bpe_vocab)
        if not pair_counts:
            print(f"No more pairs to merge at step {i}. Stopping early.")
            break

        # pick the most frequent pair (tie-break alphabetically for reproducibility)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        bpe_vocab = merge_pair(bpe_vocab, best_pair)

        if (i + 1) % 100 == 0:
            print(
                f"  merge {i + 1}/{n_merges}: {best_pair[0]!r} + {best_pair[1]!r} "
                f"= {best_pair[0] + best_pair[1]!r} (freq {pair_counts[best_pair]:,})"
                )

    return merges, sorted(base_chars)


# ── Vocab Builder ─────────────────────────────────────────────────────────────

def build_vocab(special_tokens, base_chars, merges):
    """
    Assign integer IDs to every token.
    Order: special tokens first, then base chars, then merge results.
    """
    vocab = {}
    idx = 0

    for tok in special_tokens:
        vocab[tok] = idx
        idx += 1

    for ch in sorted(base_chars):
        vocab[ch] = idx
        idx += 1

    for pair in merges:
        merged = pair[0] + pair[1]
        if merged not in vocab:
            vocab[merged] = idx
            idx += 1

    return vocab


# ── Encoder (for verification) ────────────────────────────────────────────────

def encode(text, merges, vocab, special_tokens):
    """
    Encode a string to a list of token IDs.
    Applies special token splitting first, then BPE merges.
    Unknown tokens map to vocab["[UNK]"].
    """
    special_set = set(special_tokens)
    unk_id = vocab.get("[UNK]", 1)
    words = pretokenize(text)

    ids = []
    for word in words:
        if word in special_set:
            ids.append(vocab.get(word, unk_id))
        else:
            # start as characters
            tokens = list(word)
            # apply merges in order
            for pair in merges:
                merged = pair[0] + pair[1]
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            for tok in tokens:
                ids.append(vocab.get(tok, unk_id))
    return ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run generate_dataset.py first.")

    # load sample
    print(f"Loading {TRAIN_LINES:,} lines from {INPUT_FILE}...")
    lines = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
            if len(lines) >= TRAIN_LINES:
                break

    print(f"Loaded {len(lines):,} lines.")

    # train
    merges, base_chars = train(lines, VOCAB_SIZE, SPECIAL_TOKENS)

    # build vocab
    vocab = build_vocab(SPECIAL_TOKENS, base_chars, merges)
    print(f"\nFinal vocab size: {len(vocab)}")

    # save
    out = {
        "vocab_size": VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
        "base_chars": base_chars,
        "merges": merges,
        "vocab": vocab,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

    # quick sanity check
    print("\nSanity check on 3 lines:")
    for line in lines[:3]:
        ids = encode(line, merges, vocab, SPECIAL_TOKENS)
        unk_id = vocab.get("[UNK]", 1)
        unks = ids.count(unk_id)
        print(f"  {line[:60]!r}...")
        print(f"  -> {len(ids)} tokens, {unks} UNK")


if __name__ == "__main__":
    main()
