import torch
import torch.nn.functional as F

from pretrain import get_data
from transformer import Transformer


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


def get_vocabulary(data):
    vocab = set(data)
    sorted_vocab = sorted(list(vocab))
    vocab_size = len(sorted_vocab)
    return sorted_vocab, vocab_size


def generate(model, context, tokenizer, max_new_tokens=100):
    d_batch, _ = context.shape
    for i in range(max_new_tokens):
        context = context[:, len(context) - model.context_length:]  # (d_batch, d_time)
        logits, _ = model(context)  # (d_batch * d_time, v)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, v)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
        index = index.view(d_batch, 1)  # (d_batch, d_time)
        print(tokenizer.decode(index[0].tolist()), end='')
        context = torch.cat((context, index), dim=1)
    print()


if __name__ == '__main__':
    # data_filename = input('dataset filename: ')
    data_filename = 'math.txt'

    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    tokenizer = Tokenizer(vocab)

    model = Transformer(vocab_size)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    for _ in range(20):
        context = tokenizer.encode(input('prompt: '))
        # context = torch.tensor([[0]])
        generate(model, context, tokenizer, max_new_tokens=500)
