import torch
import torch.nn.functional as F
import os

from MyGPT.vocab import Tokenizer, get_vocabulary
from MyGPT.transformer import Transformer


def generate_next_token(model, context, tokenizer):
    d_batch, _ = context.shape
    logits, _ = model(context)  # (d_batch * d_time, vocab_size)
    probs = F.softmax(logits, dim=1)  # (d_batch * d_time, vocab_size)
    index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
    index = index.view(d_batch, 1)  # (d_batch, d_time)
    next_token = tokenizer.decode(index[0].tolist())
    return next_token, index


def generate(model, context, tokenizer, num_new_tokens=500):
    for _ in range(num_new_tokens):
        context = context[:, len(context) - model.context_length:]  # (d_batch, d_time)
        next_token, index = generate_next_token(model, context, tokenizer)
        print(next_token, end="")
        context = torch.cat((context, index), dim=1)
    print()


def generate_from_pretrained(data_filename, num_prompts=20, num_tokens=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = os.path.join("weights", "vocab.json")
    vocab, vocab_size = get_vocabulary(vocab_path, data_filename)
    tokenizer = Tokenizer(vocab)

    mygpt = Transformer(vocab_size, device)
    mygpt.to(device)

    weights_path = os.path.join("weights", data_filename + ".pth")
    mygpt.load_state_dict(torch.load(weights_path))

    for _ in range(num_prompts):
        prompt = tokenizer.encode(input("PROMPT: "))
        prompt = torch.tensor(prompt, device=device).unsqueeze(0)
        generate(mygpt, prompt, tokenizer, num_new_tokens=num_tokens)
