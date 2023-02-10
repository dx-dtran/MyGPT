import torch
import torch.nn.functional as F
from transformer import Transformer
from pretrain import get_data, get_vocabulary, decode


def generate(model, indices, max_new_tokens=100):
    d_batch, d_time = indices.shape
    result = torch.zeros(1, max_new_tokens)
    for i in range(max_new_tokens):
        indices = indices[:, len(indices) - model.context_length:]  # (d_batch, d_time)
        logits, _ = model(indices)  # (d_batch * d_time, v)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, v)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, v)
        index = index.view(d_batch, d_time)  # (d_batch, d_time)
        indices = torch.cat((indices, index), dim=1)
        result[:, i] = index
    return result


if __name__ == '__main__':
    data_filename = input('dataset filename: ')

    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    context_length = 64
    model = Transformer(vocab_size, context_length)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    context = torch.tensor([[0]])
    decode(generate(model, context, max_new_tokens=2000)[0].tolist(), vocab)
