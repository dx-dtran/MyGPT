import torch

from sol import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel
from oracle import Head as MyH, MultiHeadAttention as MyMulti, FeedForward, Block as MyB, Transformer


def test_head():
    torch.manual_seed(1337)
    ak_head = Head(4)

    torch.manual_seed(1337)
    my_head = MyH(4)

    input_data = torch.rand(3, 6, 16)

    expected = ak_head(input_data)
    actual = my_head(input_data)

    print(expected.shape)
    print(actual.shape)
    assert torch.allclose(actual, expected)


def test_multihead():
    torch.manual_seed(1337)
    ak_multi = MultiHeadAttention(4, 4)

    torch.manual_seed(1337)
    my_multi = MyMulti(4, 4)

    input_data = torch.rand(3, 6, 16)

    expected = ak_multi(input_data)
    actual = my_multi(input_data)

    print(expected.shape)
    print(actual.shape)

    assert torch.allclose(actual, expected)


def test_feedforward():
    torch.manual_seed(1337)
    ak_feedforward = FeedFoward(16)

    torch.manual_seed(1337)
    my_feedforward = FeedForward(16)

    input_data = torch.rand(3, 6, 16)

    expected = ak_feedforward(input_data)
    actual = my_feedforward(input_data)

    print(expected.shape)
    print(actual.shape)

    assert torch.allclose(actual, expected)


def test_block():
    torch.manual_seed(1337)
    ak_block = Block(16, 4)

    torch.manual_seed(1337)
    my_block = MyB(16, 4)

    input_data = torch.rand(3, 6, 16)

    expected = ak_block(input_data)
    actual = my_block(input_data)

    print(expected.shape)
    print(actual.shape)

    assert torch.allclose(actual, expected)


def test_transformer():
    torch.manual_seed(1337)
    ak_transformer = GPTLanguageModel()

    torch.manual_seed(1337)
    my_transformer = Transformer()

    input_idx = torch.randint(0, 128, (4, 6))
    targets = torch.randint(0, 128, (4, 6))

    expected, expected_loss = ak_transformer(input_idx, targets)
    actual, actual_loss = my_transformer(input_idx, targets)

    print(expected.shape)
    print(actual.shape)

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_loss, expected_loss)


if __name__ == '__main__':
    test_head()
    test_multihead()
    test_feedforward()
    test_block()
    test_transformer()
