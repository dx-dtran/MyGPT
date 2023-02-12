import torch
import os
from transformer import Transformer
from generate import generate
from vocab import Tokenizer, create_vocabulary, save_vocabulary


def get_data(filename):
    try:
        with open(filename, 'r') as input_file:
            input_data = input_file.read()
            return input_data
    except FileNotFoundError:
        print('data file not found')


def get_train_val_data(tokenizer, data, train_val_split=0.9):
    data_tensor = tokenizer.encode(data)
    n = int(data_tensor.shape[1] * train_val_split)
    train_data = data_tensor[0][:n]
    val_data = data_tensor[0][n:]
    return train_data, val_data


def get_batch(data, batch_size, context_length):
    x = []
    y = []
    for i in range(batch_size):
        index = torch.randint(0, len(data) - context_length - 1, (1,))
        x.append(data[index:index + context_length])
        y.append(data[index + 1:index + context_length + 1])
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


def estimate_loss(model, data, batch_size, context_length, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for iteration in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length)
        _, loss = model(x, y)
        losses[iteration] = loss
    model.train()
    return losses.mean()


if __name__ == '__main__':
    # data_filename = input('dataset filename: ')
    data_filename = 'math.txt'

    torch.manual_seed(3)

    data_path = os.path.join('data', data_filename)
    raw_data = get_data(data_path)
    vocab, vocab_size = create_vocabulary(raw_data)

    vocab_path = os.path.join('weights', 'vocab.json')
    save_vocabulary(vocab_path, data_filename, vocab)

    tokenizer = Tokenizer(vocab)
    train_data, val_data = get_train_val_data(tokenizer, raw_data)

    device = 'cpu'
    batch_size = 16
    max_iters = 5000
    eval_interval = 100
    eval_iters = 100
    learning_rate = 1e-3

    context_length = 64

    mygpt = Transformer(vocab_size, context_length=context_length, d_embed=32, n_head=4, n_layer=4)
    mygpt.to(device)

    num_params = sum(param.numel() for param in mygpt.parameters())
    print('mygpt {} parameter model initialized'.format(num_params))

    optimizer = torch.optim.AdamW(mygpt.parameters(), lr=learning_rate)

    print('begin training using {}'.format(device))
    for iteration in range(max_iters):
        if iteration == 0 or iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = estimate_loss(mygpt, train_data, batch_size, context_length, eval_iters)
            val_loss = estimate_loss(mygpt, val_data, batch_size, context_length, eval_iters)
            print("\n================================================================")
            print(
                "iteration: {} | training loss: {:.3f} | validation loss: {:.3f}".format(
                    iteration, train_loss, val_loss
                )
            )
            print("================================================================\n")
            context = torch.tensor([[0]])
            generate(mygpt, context, tokenizer, num_new_tokens=200)

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = mygpt(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # weights_path = os.path.join('weights', data_filename + '.pth')
    # torch.save(mygpt.state_dict(), weights_path)
