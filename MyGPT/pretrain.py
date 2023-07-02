import torch
import os
import time
import tiktoken
import numpy as np

from MyGPT.transformer import Transformer
from MyGPT.generate import generate
from MyGPT.vocab import Tokenizer, create_vocabulary


def get_data(filename):
    try:
        with open(filename, "r") as input_file:
            input_data = input_file.read()
            return input_data
    except FileNotFoundError:
        print("data file not found")


def get_train_val_data(data, device, train_val_split=0.9):
    enc = tiktoken.get_encoding("gpt2")

    def encode(s):
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    encoded_data = encode(data)
    data_tensor = torch.tensor(encoded_data, device=device).unsqueeze(0)
    n = int(data_tensor.shape[1] * train_val_split)
    train_data = data_tensor[0][:n]
    val_data = data_tensor[0][n:]
    return train_data, val_data


# def get_batch(data, batch_size, context_length):
#     x, y = [], []
#     for i in range(batch_size):
#         index = torch.randint(0, len(data) - context_length - 1, (1,))
#         x.append(data[index: index + context_length])
#         y.append(data[index + 1: index + context_length + 1])
#     x, y = torch.stack(x), torch.stack(y)
#     return x, y


def get_batch(split, batch_size, context_length):
    train_dir = os.path.join("data", "train.bin")
    val_dir = os.path.join("data", "val.bin")
    train_data = np.memmap(train_dir, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_dir, dtype=np.uint16, mode='r')
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + context_length]).astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, split, batch_size, context_length, device, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    for iteration in range(eval_iters):
        x, y = get_batch(split, batch_size, context_length)
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses[iteration] = loss
    model.train()
    return losses.mean()


def pretrain():
    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # obtain the dataset
    # data_path = os.path.join("data", data_filename)
    # raw_data = get_data(data_filename)

    # create the vocabulary
    # vocab, vocab_size = create_vocabulary(raw_data)
    # tokenizer = Tokenizer(vocab)

    # convert the raw data to tensors
    # train_data, val_data = get_train_val_data(raw_data, tokenizer, device)

    # define the training hyperparameters
    batch_size = 16
    max_iters = 5000
    eval_interval = 250
    eval_iters = 100
    learning_rate = 1e-3
    context_length = 128

    # define the model
    mygpt = Transformer(
        50304,
        device,
        context_length=context_length,
        d_embed=128,
        n_head=2,
        n_layer=2,
    )
    mygpt.to(device)

    num_params = sum(param.numel() for param in mygpt.parameters())
    print("MyGPT initialized with {} parameters".format(num_params))
    print("Begin training using {}".format(device))

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    start = time.time()
    for iteration in range(max_iters):
        if iteration != 0 and iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = estimate_loss(
                mygpt, "train", batch_size, context_length, device, eval_iters
            )
            val_loss = estimate_loss(
                mygpt, "val", batch_size, context_length, device, eval_iters
            )

            print("\n===========================================================================================")
            print(
                "iteration: {} | training loss: {:0.3f} | validation loss: {:0.3f} | elapsed: {:0.2f} seconds ".format(
                    iteration, train_loss, val_loss, time.time() - start
                )
            )
            print("===========================================================================================\n")

            # context = torch.tensor([[0]], dtype=torch.long, device=device)
            # generate(mygpt, context, num_new_tokens=200)

        iter_time = time.time()

        x, y = get_batch("train", batch_size, context_length)
        x = x.to(device)
        y = y.to(device)
        _, loss = mygpt(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("iter: {}, loss: {:0.3f}, time: {:0.3f}s".format(iteration, loss, time.time() - iter_time))

    print("Total training time: {:0.2f} seconds".format(time.time() - start))

    # save the model weights

    weights_path = os.path.join("weights", "tinystories" + ".pth")
    torch.save(mygpt.state_dict(), weights_path)
