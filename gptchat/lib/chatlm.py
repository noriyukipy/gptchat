import torch
import transformers


class ChatLMTokenizerBuilder:
    def build(self):
        # Prepare tokenizer
        cls = "cl-tohoku/bert-base-japanese"
        tokenizer = transformers.AutoTokenizer.from_pretrained(cls)

        # Add special tokens
        special_tokens_dict = {
            "additional_special_tokens": [
                "<CTX>",  # Context symbol for token_type_ids
                "<RES>",  # Response symbol for token_type_ids
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer


class ChatLMModelInputBuilder():
    def __init__(self, tokenizer, add_end_token):
        self._tokenizer = tokenizer
        self._add_end_token = add_end_token

    def build(self, ctx, res, batch_size=None):
        CTX, RES = self._tokenizer.additional_special_tokens

        if self._add_end_token:
            end_token = [self._tokenizer.cls_token]
        else:
            end_token = []
        tokens = (
            self._tokenizer.tokenize(ctx),
            [self._tokenizer.sep_token] + self._tokenizer.tokenize(res) + end_token
        )

        # Build input ids
        input_ids = self._tokenizer.convert_tokens_to_ids(sum(tokens, []))

        # Build token type ids
        token_types = [CTX] * len(tokens[0]) + [RES] * len(tokens[1])
        token_type_ids = self._tokenizer.convert_tokens_to_ids(token_types)

        # Build target ids
        border_idx = len(tokens[0])
        target_ids = [-100] * border_idx + input_ids[border_idx:]

        assert len(input_ids) == len(token_type_ids) == len(target_ids)

        if batch_size:
            input_ids = [input_ids for _ in range(batch_size)]
            token_type_ids = [token_type_ids for _ in range(batch_size)]
            target_ids = [target_ids for _ in range(batch_size)]

        return dict(
            input_ids=torch.tensor(input_ids),
            token_type_ids=torch.tensor(token_type_ids),
            target_ids=torch.tensor(target_ids),
        )

    def update(self, model_input, next_ids):
        # [TODO] Implement target inputs
        input_ids = model_input["input_ids"]
        token_type_ids = model_input["token_type_ids"]
        batch_size = int(input_ids.size()[0])

        CTX_id, RES_id = self._tokenizer.additional_special_tokens_ids
        input_ids = torch.cat([input_ids, next_ids], dim=-1)
        token_type_ids = torch.cat(
            [token_type_ids,
             torch.tensor([[RES_id] for _ in range(batch_size)])],
            dim=1
        )
        return dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )

    def ended(self, model_input):
        input_ids = model_input["input_ids"]
        cond = (input_ids == self._tokenizer.cls_token_id).sum(dim=1) > 0
        return bool(torch.all(cond))


class ChatLMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, ctx_res_list):
        input_builder = ChatLMModelInputBuilder(
            tokenizer=tokenizer,
            add_end_token=False,
        )
        input_ids_list = []
        token_type_ids_list = []
        target_ids_list = []

        for ctx, res in ctx_res_list:
            model_input = input_builder.build(ctx, res, batch_size=None)
            input_ids_list.append(model_input["input_ids"])
            token_type_ids_list.append(model_input["token_type_ids"])
            target_ids_list.append(model_input["target_ids"])

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
                padding_value=self._padding_value if idx <= 1 else -100
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
