import torch
from gptchat.lib import generator


def test_filter_to_topk():
    res = generator.filter_to_topk(
        top_k=2,
        dist=torch.Tensor([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]])
    )

    inf = float("Inf")
    expected = torch.Tensor(
        [[2, -inf, 3, -inf, -inf],
         [-inf, -inf, -inf, 8, 9]]
    )
    assert torch.all(torch.eq(res, expected))


def test_filter_to_topk_out_of_index():
    res = generator.filter_to_topk(
        top_k=10,
        dist=torch.Tensor([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]])
    )
    expected = torch.Tensor(
        [[2, 0, 3, 1, -1],
         [5, 6, 7, 8, 9]]
    )
    assert torch.all(torch.eq(res, expected))


def test_filter_to_topp():
    res = generator.filter_to_topp(
        top_p=0.9,
        dist=torch.Tensor([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]])
    )
    inf = float("Inf")
    expected = torch.Tensor(
        [[2, -inf, 3, 1, -inf],
         [-inf, -inf, 7, 8, 9]]
    )
    assert torch.all(torch.eq(res, expected))


def test_filter_to_topp_under():
    res = generator.filter_to_topp(
        top_p=0,
        dist=torch.Tensor([[2, 0, 3, 1, -1], [5, 6, 7, 8, 9]])
    )
    inf = float("Inf")
    expected = torch.Tensor(
        [[-inf, -inf, 3, -inf, -inf],
         [-inf, -inf, -inf, -inf, 9]]
    )
    assert torch.all(torch.eq(res, expected))
