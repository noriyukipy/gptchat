import torch


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, block_size):
        ids = []
        for text in texts:
            # Set add_special_tokens=False
            # not to add additional special tokens.
            tokens = tokenizer.tokenize(text)
            ids.extend(tokenizer.convert_tokens_to_ids(tokens))

        inputs = []
        for idx in range(0, len(ids)-block_size+1, block_size):
            inputs.append(ids[idx:idx+block_size])

        self._inputs = inputs

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, item):
        return torch.tensor(self._inputs[item])


def build_pretrained_data(
    tokenizer, texts, batch_size, block_size, shuffle
):
    """Prepare DataSet and DataLoader."""
    data_set = LMDataset(tokenizer, texts, block_size=block_size)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_set, data_loader
