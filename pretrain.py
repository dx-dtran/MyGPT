import torch
from transformer import Transformer


def get_data(filename):
    with open(filename, 'r') as input_file:
        input_data = input_file.read()
    return input_data


def get_vocabulary(data):
    vocab = set(data)
    sorted_vocab = sorted(list(vocab))
    vocab_size = len(sorted_vocab)
    return sorted_vocab, vocab_size


def encode(data, vocab):
    atoi = {char: i for i, char in enumerate(vocab)}
    return [atoi[char] for char in data]


def get_train_val_data(data, vocab):
    data_tensor = torch.tensor(encode(data, vocab))
    n = int(len(data_tensor) * 0.9)
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
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
    data_filename = input('dataset filename: ')

    torch.manual_seed(3)
    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)
    train_data, val_data = get_train_val_data(data, vocab)

    device = 'cpu'
    batch_size = 16
    max_iters = 5000
    eval_interval = 100
    eval_iters = 200
    learning_rate = 1e-3

    context_length = 64

    model = Transformer(vocab_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(max_iters):
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = estimate_loss(model, train_data, batch_size, context_length, eval_iters)
            val_loss = estimate_loss(model, val_data, batch_size, context_length, eval_iters)
            print(
                "iteration: {} training loss: {:.3f} validation loss: {:.3f}".format(
                    iteration, train_loss, val_loss
                )
            )

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'weights/{}.pth'.format(data_filename))
