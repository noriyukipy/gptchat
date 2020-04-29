import torch


def extract_response_tokens(tokens, start_token, end_token):
    """
    Args:
        tokens (List[str]): tokens consisting of context and response.
            response starts with start_token, and possibly ends with end_token
        start_token (str): Token from which
    """
    start_idx = tokens.index(start_token)
    try:
        end_idx = tokens.index(end_token)
    except ValueError:
        end_idx = len(tokens) - 1
    return tokens[start_idx+1:end_idx]


def extract_response_ids(input_ids, start_token_id, end_token_id):
    batch_size = input_ids.size()[0]
    start_index = [float("infinity") for _ in range(batch_size)]
    eos_index = [float("infinity") for _ in range(batch_size)]

    # Update EOS token index
    for sample_idx, idx in torch.nonzero(input_ids == start_token_id):
        idx = int(idx)
        if idx < start_index[sample_idx]:
            start_index[sample_idx] = idx

    # Update EOS token index
    for sample_idx, idx in torch.nonzero(input_ids == end_token_id):
        idx = int(idx)
        if idx < eos_index[sample_idx]:
            eos_index[sample_idx] = idx

    # if eos index is infinity, set it to the last index
    eos_index = [
        input_ids.size()[1]-1 if x == float("infinity") else x
        for x in eos_index
    ]

    return [ids[start_index[idx]+1:eos_index[idx]] for idx, ids in enumerate(input_ids)]
