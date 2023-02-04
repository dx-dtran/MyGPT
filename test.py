import torch

from do_not_look import Head, MultiHeadAttention
from oracle import Head as MyH


def test_head():
    ak_head = Head(4)
    # my_head = MyH(4)
    input_data = torch.rand(3, 6, 5)

    expected = ak_head(input_data)
    # actual = my_head(input_data)
    # assert torch.allclose(expected, actual)


if __name__ == '__main__':
    test_head()
