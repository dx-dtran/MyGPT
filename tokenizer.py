import torch


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
        return ''.join(itoa[index] for index in indices)


def get_data(filename):
    with open(filename, 'r') as input_file:
        input_data = input_file.read()
    return input_data


def get_vocabulary(data):
    vocab = set(data)
    sorted_vocab = sorted(list(vocab))
    vocab_size = len(sorted_vocab)
    return sorted_vocab, vocab_size
