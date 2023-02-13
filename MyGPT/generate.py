import torch
import torch.nn.functional as F
import os

from vocab import Tokenizer, get_vocabulary
from transformer import Transformer


def generate(model, context, tokenizer, num_new_tokens=500):
    d_batch, _ = context.shape
    for i in range(num_new_tokens):
        context = context[:, len(context) - model.context_length :]  # (d_batch, d_time)
        logits, _ = model(context)  # (d_batch * d_time, vocab_size)
        probs = F.softmax(logits, dim=1)  # (d_batch * d_time, vocab_size)
        index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
        index = index.view(d_batch, 1)  # (d_batch, d_time)
        print(tokenizer.decode(index[0].tolist()), end="")
        context = torch.cat((context, index), dim=1)
    print()


if __name__ == "__main__":
    # data_filename = input('dataset filename: ')
    data_filename = "math.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = os.path.join("..", "weights", "vocab.json")
    vocab, vocab_size = get_vocabulary(vocab_path, data_filename)
    tokenizer = Tokenizer(vocab, device)

    mygpt = Transformer(vocab_size, device)

    weights_path = os.path.join("..", "weights", data_filename + ".pth")
    mygpt.load_state_dict(torch.load(weights_path))

    for _ in range(20):
        prompt = tokenizer.encode(input("PROMPT: "))
        # prompt = torch.tensor([[0]], device=device)
        generate(mygpt, prompt, tokenizer, num_new_tokens=2000)
