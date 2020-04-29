import torch


def split_context_response_ids(input_ids, start_token_id, end_token_id):
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

    return [
        (ids[:start_index[idx]], ids[start_index[idx]+1:eos_index[idx]])
        for idx, ids in enumerate(input_ids)
    ]


def filter_out_by_coverage_ratio(context_response_ids, ratio):
    """
    Args:
        context_response_ids: (Tuple[List[int], List[int]])
        ratio: float
    """
    filtered_cands = []
    for context_ids, response_ids in context_response_ids:
        # Filter out response with length 0
        if len(response_ids) == 0:
            continue

        num_total = 0
        num_overwrap = 0
        for rid in response_ids:
            num_total += 1
            if rid in context_ids:
                num_overwrap += 1
        if num_overwrap / num_total > ratio:
            continue
        filtered_cands.append((context_ids, response_ids))
   
    if not filtered_cands:
        return context_response_ids

    return filtered_cands
