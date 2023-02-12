import torch
import json


def create_vocabulary(data):
    vocab = set(data)
    sorted_vocab = sorted(list(vocab))
    vocab_size = len(sorted_vocab)
    return sorted_vocab, vocab_size


def save_vocabulary(filename, vocab_id, vocab):
    try:
        with open(filename, "r") as vocab_file:
            vocab_dict = json.load(vocab_file)
    except FileNotFoundError:
        vocab_dict = {}

    if vocab_id not in vocab_dict:
        obj = {"vocab": vocab, "vocab_size": len(vocab)}
        vocab_dict[vocab_id] = obj
        with open(filename, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
            print("new vocab has been saved")
    else:
        print("vocab has already been saved")


def get_vocabulary(vocab_filename, vocab_id):
    try:
        with open(vocab_filename, "r") as vocab_file:
            vocab_dict = json.load(vocab_file)
            return vocab_dict[vocab_id]["vocab"], vocab_dict[vocab_id]["vocab_size"]
    except FileNotFoundError:
        print("vocab file not found")
    except KeyError:
        print("vocab id not found in file")


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def encode(self, words):
        atoi = {char: i for i, char in enumerate(self.vocab)}
        encoded_chars = [atoi[char] for char in words]
        return torch.tensor(encoded_chars).unsqueeze(0)

    def decode(self, indices):
        itoa = {i: char for i, char in enumerate(self.vocab)}
        return "".join(itoa[index] for index in indices)
