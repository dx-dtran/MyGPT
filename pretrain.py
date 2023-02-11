import torch
from transformer import Transformer
from generate import generate
from tokenizer import Tokenizer, get_data, get_vocabulary


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
    data_filename = input('dataset filename: ')

    torch.manual_seed(3)
    data = get_data('data/{}'.format(data_filename))
    vocab, vocab_size = get_vocabulary(data)

    tokenizer = Tokenizer(vocab)
    train_data, val_data = get_train_val_data(tokenizer, data)

    device = 'cpu'
    batch_size = 16
    max_iters = 5000
    eval_interval = 100
    eval_iters = 100
    learning_rate = 1e-3

    context_length = 64

    model = Transformer(vocab_size, context_length=context_length, d_embed=32, n_head=4, n_layer=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(max_iters):
        if iteration == 0 or iteration % eval_interval == 0 or iteration == max_iters - 1:
            train_loss = estimate_loss(model, train_data, batch_size, context_length, eval_iters)
            val_loss = estimate_loss(model, val_data, batch_size, context_length, eval_iters)
            print("============================================================")
            print(
                "iteration: {} training loss: {:.3f} validation loss: {:.3f}".format(
                    iteration, train_loss, val_loss
                )
            )
            print("============================================================", end='\n')
            context = torch.tensor([[0]])
            generate(model, context, tokenizer, max_new_tokens=200)

        x, y = get_batch(train_data, batch_size, context_length)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # torch.save(model.state_dict(), 'weights/{}.pth'.format(data_filename))
