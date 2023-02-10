import torch
import torch.nn.functional as F
from transformer import Transformer
from pretrain import get_data, get_vocabulary


def generate(model, indices, max_new_tokens=100):
    d_batch, _ = indices.shape
    for i in range(max_new_tokens):
        indices = indices[:, len(indices) - model.context_length:]  # (d_batch, d_time)
        logits, _ = model(indices)  # (d_batch * d_time, v)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, v)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
        index = index.view(d_batch, 1)  # (d_batch, d_time)
        print(decode(index.item(), vocab), end='')
        indices = torch.cat((indices, index), dim=1)


def tokenize(data, vocab):
    atoi = {char: i for i, char in enumerate(vocab)}
    l = [atoi[char] for char in data]
    return torch.tensor(l).unsqueeze(0)


def decode(index, vocab):
    itoa = {i: char for i, char in enumerate(vocab)}
    return itoa[index]


if __name__ == '__main__':
    # data_filename = input('dataset filename: ')
    data_filename = 'shakespeare.txt'

    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    model = Transformer(vocab_size)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    # context = tokenize(input('prompt: '), vocab)
    context = torch.tensor([[0]])
    generate(model, context, max_new_tokens=20000)
