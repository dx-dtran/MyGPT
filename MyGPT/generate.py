import torch
import torch.nn.functional as F

from vocab import Tokenizer, get_vocabulary
from transformer import Transformer


def generate(model, prompt, tokenizer, max_new_tokens=500):
    d_batch, _ = prompt.shape
    for i in range(max_new_tokens):
        prompt = prompt[:, len(prompt) - model.context_length:]  # (d_batch, d_time)
        logits, _ = model(prompt)  # (d_batch * d_time, vocab_size)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, vocab_size)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
        index = index.view(d_batch, 1)  # (d_batch, d_time)
        print(tokenizer.decode(index[0].tolist()), end='')
        prompt = torch.cat((prompt, index), dim=1)
    print()


if __name__ == '__main__':
    # data_filename = input('dataset filename: ')
    data_filename = 'math.txt'

    vocab, vocab_size = get_vocabulary('weights/vocab.json', data_filename)

    tokenizer = Tokenizer(vocab)

    model = Transformer(vocab_size)
    model.load_state_dict(torch.load('weights/{}.pth'.format(data_filename)))

    for _ in range(20):
        prompt = tokenizer.encode(input('prompt: '))
        # context = torch.tensor([[0]])
        generate(model, prompt, tokenizer)
