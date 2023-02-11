import torch
import torch.nn.functional as F
from transformer import Transformer
from tokenizer import Tokenizer, get_data, get_vocabulary


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
