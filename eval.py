import torch
from mygpt import Transformer
from train import get_data, get_vocabulary, decode

if __name__ == '__main__':
    data_filename = input('dataset filename: ')

    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    context_length = 32
    model = Transformer(vocab_size, context_length)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    context = torch.tensor([[0]])
    decode(model.generate(context, max_new_tokens=10000)[0].tolist(), vocab)
