import torch
import torch.nn.functional as F

from vocab import Tokenizer, get_vocabulary
from transformer import Transformer


def generate(model, context, tokenizer, num_new_tokens=500):
    d_batch, _ = context.shape
    for i in range(num_new_tokens):
        context = context[:, len(context) - model.context_length:]  # (d_batch, d_time)
        logits, _ = model(context)  # (d_batch * d_time, vocab_size)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, vocab_size)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
        index = index.view(d_batch, 1)  # (d_batch, d_time)
        print(tokenizer.decode(index[0].tolist()), end='')
        context = torch.cat((context, index), dim=1)
    print()


if __name__ == '__main__':
    # data_filename = input('dataset filename: ')
    data_filename = 'calculus.txt'

    # todo: use os lib to build the file path
    vocab, vocab_size = get_vocabulary('weights/vocab.json', data_filename)

    tokenizer = Tokenizer(vocab)

    mygpt = Transformer(vocab_size)
    mygpt.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    for _ in range(20):
        prompt = tokenizer.encode(input('PROMPT: '))
        # prompt = torch.tensor([[0]])
        generate(mygpt, prompt, tokenizer, num_new_tokens=5000)
