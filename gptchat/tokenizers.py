class TokenizerWrapper:
    """TokenizerWrapper wraps libraries given by HuggingFace tokenizers
    to adopt interface of HuggingFace transformers tokenizer.
    """

    sep_token = "[SEP]"
    cls_token = "[CLS]"
    pad_token = "[PAD]"

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, *args, **kwargs):
        return self._tokenizer.encode(*args, **kwargs).ids

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    @property
    def sep_token_id(self):
        return self._tokenizer.get_vocab()[self.sep_token]

    @property
    def cls_token_id(self):
        return self._tokenizer.get_vocab()[self.cls_token]

    @property
    def pad_token_id(self):
        return self._tokenizer.get_vocab()[self.pad_token]

    def __len__(self):
        return self._tokenizer.get_vocab_size()
