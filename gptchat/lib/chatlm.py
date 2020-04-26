import torch
import transformers


class ChatLMTokenizerBuilder:
    def build(self):
        # Prepare tokenizer
        cls = "bert-base-japanese"
        tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(cls)

        # Add special tokens
        special_tokens_dict = {
            "additional_special_tokens": [
                "<CTX>",  # Context symbol for token_type_ids
                "<RES>",  # Response symbol for token_type_ids
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer


class ChatLMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, ctx_res_list):
        CTX, RES = tokenizer.additional_special_tokens
        input_ids_list = []
        token_type_ids_list = []
        target_ids_list = []
        for ctx, res in ctx_res_list:
            tokens = (
                tokenizer.tokenize(ctx),
                [tokenizer.sep_token] + tokenizer.tokenize(res) + [tokenizer.cls_token]
            )

            # Build input ids
            input_ids = tokenizer.convert_tokens_to_ids(sum(tokens, []))

            # Build token type ids
            token_types = [CTX] * len(tokens[0]) + [RES] * len(tokens[1])
            token_type_ids = tokenizer.convert_tokens_to_ids(token_types)

            # Build target ids
            border_idx = len(tokens[0])
            target_ids = [-100] * border_idx + input_ids[border_idx:]

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            target_ids_list.append(target_ids)

        self._input_ids_list = input_ids_list
        self._token_type_ids_list = token_type_ids_list
        self._target_ids_list = target_ids_list

    def __len__(self):
        return len(self._input_ids_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self._input_ids_list[idx]),
            torch.tensor(self._token_type_ids_list[idx]),
            torch.tensor(self._target_ids_list[idx]),
        )


class ChatLMCollation:
    def __init__(self, padding_value):
        self._padding_value = padding_value

    def apply(self, batch):
        paired_batch = list(zip(*batch))

        res = []
        for idx, items in enumerate(paired_batch):
            x = torch.nn.utils.rnn.pad_sequence(
                items,
                batch_first=True,
                padding_value=self._padding_value
            )
            res.append(x.reshape(len(batch), -1, x.size()[1]))

        return {
            "input_ids": res[0],
            "token_type_ids": res[1],
            "labels": res[2],
        }


class ChatLMDataloaderBuilder:
    def build(self, dataset, batch_size, shuffle, pad_token_id):
        collate_fn = ChatLMCollation(pad_token_id).apply
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        return data_loader


class ChatLMModelBuilder:
    def from_pretrained(self, pretrained_dir, vocab_size):
        net = transformers.GPT2LMHeadModel.from_pretrained(pretrained_dir)
        net.resize_token_embeddings(vocab_size)
        return net
