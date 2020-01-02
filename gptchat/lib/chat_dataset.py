from gptchat.lib import special_tokens
import torch


class DialogDataset(torch.utils.data.Dataset):
    """
    [TODO] Special Token を辞書に追加する
    """
    def __init__(self, tokenizer, inputs):
        self._tokenizer = tokenizer
        inputs_ = []
        for text, reply, distructors in inputs:
            item = self._build(text, reply, distructors)
            inputs_.append(item)

        self._inputs = inputs_

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return self._inputs[idx]
        # ipt = self._inputs[idx]
        # token_ids, segment_ids, target_ids = self._transform(ipt)
        # print(token_ids)
        # return token_ids, segment_ids, target_ids

    def _build(self, text, reply, distructors):
        # list to be returned
        token_ids_ = []
        segment_ids_ = []
        target_ids_ = []
        mc_ids_ = []

        for idx, rpl in enumerate([reply] + distructors):
            is_distructor = idx > 0  # distructor flag
            token_ids, segment_ids, target_ids, mc_id = self._build_one(
                text,
                reply,
                is_distructor
            )
            assert len(token_ids) == len(segment_ids) == len(target_ids)

            token_ids_.append(token_ids)
            segment_ids_.append(segment_ids)
            target_ids_.append(target_ids)
            mc_ids_.append(mc_id)

        return token_ids_, segment_ids_, target_ids_, mc_ids_, 0

    def _build_one(self, text, reply, is_distructor):
        tokenizer = self._tokenizer

        seq = [
            [special_tokens.BOS] + tokenizer.tokenize(text),
            [special_tokens.SEP] + tokenizer.tokenize(reply) + [special_tokens.EOS],
        ]
        # Build tokens
        tokens = sum(seq, [])
        # Build ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Build segment
        # [TODO] Consider there are any problems to use same id special_tokens.SP2 in token and segment...
        segments = (
            [special_tokens.SP1] * len(seq[0]) +
            [special_tokens.SP2] * len(seq[1])
        )
        segment_ids = tokenizer.convert_tokens_to_ids(segments)

        # Build target
        # target labels are shifted by GPT2 models
        # - https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2DoubleHeadsModel

        target_ids = [-1] * len(seq[0])
        # Ignore all target_ids if reply is a distractor
        if is_distructor:
            target_ids += [-1] * len(seq[1])
        else:
            target_ids += [-1] + token_ids[len(seq[0])+1:]
        # mc_id is a position of last token.
        mc_id = len(token_ids) - 1

        return token_ids, segment_ids, target_ids, mc_id


class PaddingCollation:
    def __init__(self, padding_value):
        self._padding_value = padding_value

    def apply(self, batch):
        """
        Returns:
            2 -> batch_size
            3 -> 1 + num_distructors
            7 -> max_length
            [
                torch.Size([2, 3, 7]),
                torch.Size([2, 3, 7]),
                torch.Size([2, 3, 7]),
                torch.Size([2, 3]),
                torch.Size([2])
            ]
        """
        lst = []

        paired_batch = list(zip(*batch))
        lm_items = paired_batch[:-2]
        mc_items = paired_batch[-2:]

        mc_ids, mc_labels = [torch.tensor(x) for x in mc_items]
        for idx, items in enumerate(lm_items):
            x = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in sum(list(items), [])],
                batch_first=True,
                padding_value=self._padding_value
            )
            lst.append(x.reshape(len(batch), -1, x.size()[1]))

        return [*lst, mc_ids, mc_labels]
