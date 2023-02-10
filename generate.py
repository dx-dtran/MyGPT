import torch
from transformer import Transformer
from pretrain import get_data, get_vocabulary, decode

if __name__ == '__main__':
    data_filename = input('dataset filename: ')

    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    context_length = 32
    # model = Transformer(vocab_size, d_embed=128, n_head=8, context_length=context_length)
    model = Transformer(vocab_size, context_length)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))
    print('loaded weights')

    context = torch.tensor([[0]])
    decode(model.generate(context, max_new_tokens=2000)[0].tolist(), vocab)
