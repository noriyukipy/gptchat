class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, *args, **kwargs):
        return self._tokenizer.encode(*args, **kwargs).ids

    @property
    def sep_token_id(self):
        return self._tokenizer.get_vocab()["[SEP]"]

    @property
    def cls_token_id(self):
        return self._tokenizer.get_vocab()["[CLS]"]

    @property
    def pad_token_id(self):
        return self._tokenizer.get_vocab()["[PAD]"]

    def __len__(self):
        return self._tokenizer.get_vocab_size()