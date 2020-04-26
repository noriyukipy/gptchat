import torch


def filter_to_topk(top_k, dist):
    """Replace all the elements in dist to -Inf
    except the top k largest elements.

    Args:
        top_k (int):
        dist (torch.Tensor): (num_batch, -1) dimensioned torch tensor.
    Returns:
        torch.Tensor: (num_batch, -1) dimensioned torch tensor.
    """
    dist = dist.clone()
    top_k = min(top_k, dist.shape[1])
    # threshold dim = (batch_size, 1)
    threshold = torch.topk(dist, top_k)[0][:, -1, None]
    dist[dist < threshold] = -float("Inf")
    return dist


def filter_to_topp(top_p, dist):
    """
    Args:
        top_k (int):
        dist (torch.Tensor): (num_batch, -1) dimensioned torch tensor.
    Returns:
        torch.Tensor: (num_batch, -1) dimensioned torch tensor.
    """
    dist = dist.clone()

    # sort for each row
    sorted_dist, sorted_idx = torch.sort(dist, descending=True)
    # cumulative probability sum for each row
    prob_dist_cusum = torch.cumsum(
        torch.nn.functional.softmax(sorted_dist, dim=-1),
        dim=-1
    )

    # Detect the filter index
    removed_index = torch.cat(
        [
            torch.tensor([[False] for _ in range(prob_dist_cusum.shape[0])]),
            prob_dist_cusum > top_p,
        ],
        dim=1
    )[:, :-1]

    # pass (dim, index, source)
    mask_flag = removed_index.scatter(1, sorted_idx, removed_index)
    dist[mask_flag] = -float("Inf")

    return dist


def filter_bad_ids(bad_ids, dist):
    dist = dist.clone()
    dist[:, bad_ids] = -float("Inf")
    return dist


def sample_multinomial(dist):
    return torch.multinomial(
        input=torch.functional.F.softmax(dist, dim=-1),
        num_samples=1
    )


class TopPKGenerator:
    """Sentence generator to sandom sampling from top-k distribution"""
    def __init__(self, model, top_p, top_k, bad_ids):
        self._model = model
        self._top_p = top_p
        self._top_k = top_k
        self._bad_ids = bad_ids

    def step(self, **argv):
        # Predict next word distribution
        output = self._model(**argv)
        # last_hidden_state dim = (batch_size, input_ids length, num_vocabs)
        last_hidden_state = output[0]
        next_id_dist = last_hidden_state[:, -1, :]

        filters = [
            lambda dist: filter_to_topp(self._top_p, dist),
            lambda dist: filter_to_topk(self._top_k, dist),
            lambda dist: filter_bad_ids(self._bad_ids, dist),
        ]
        filtered_dist = next_id_dist
        for flt in filters:
            filtered_dist = flt(filtered_dist)

        return sample_multinomial(filtered_dist), output
