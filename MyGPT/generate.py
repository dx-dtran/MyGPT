import torch
import torch.nn.functional as F
import os
import tiktoken

from MyGPT.vocab import Tokenizer, get_vocabulary
from MyGPT.transformer import Transformer


def generate_next_token(model, context):

    def decode(arr):
        enc = tiktoken.get_encoding("gpt2")
        return enc.decode(arr)

    d_batch, _ = context.shape
    scores, _ = model(context)  # (d_batch * d_time, vocab_size)
    probs = F.softmax(scores, dim=1)  # (d_batch * d_time, vocab_size)
    index = torch.multinomial(probs[-1], 1)  # (d_batch * d_time, 1)
    index = index.view(d_batch, 1)  # (d_batch, d_time)
    next_token = decode(index[0].tolist())
    return next_token, index


def generate(model, context, num_new_tokens=500):
    for _ in range(num_new_tokens):
        context = context[:, len(context) - model.context_length:]  # (d_batch, d_time)
        next_token, index = generate_next_token(model, context)
        print(next_token, end="")
        context = torch.cat((context, index), dim=1)
    print()


def generate_from_pretrained(data_filename, num_prompts=20, num_tokens=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # vocab_path = os.path.join("weights", "vocab.json")
    # vocab, vocab_size = get_vocabulary(vocab_path, data_filename)
    # tokenizer = Tokenizer(vocab)

    mygpt = Transformer(50304, device, context_length=512, d_embed=768, n_head=16, n_layer=8)
    mygpt.to(device)

    weights_path = os.path.join("weights", data_filename + ".pth")
    mygpt.load_state_dict(torch.load(weights_path))

    def encode(arr):
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode(arr, allowed_special={"<|endoftext|>"})

    for _ in range(num_prompts):
        prompt = encode(input("PROMPT: "))
        prompt = torch.tensor(prompt, device=device).unsqueeze(0)
        generate(mygpt, prompt, num_new_tokens=num_tokens)
