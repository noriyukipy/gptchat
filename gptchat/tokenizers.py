import sentencepiece as spm


class SentencePieceTokenizer:
    """Tokenizer with SentencePiece"""

    sep_token = "[SEP]"
    cls_token = "[CLS]"
    pad_token = "[PAD]"

    def __init__(self):
        # This attribute is set after loading model
        self._spm = None

    def train(self, **kwargs):
        spm.SentencePieceTrainer.train(**kwargs)

    def load(self, *args, **kwargs):
        self._spm = spm.SentencePieceProcessor()
        self._spm.load(*args, **kwargs)
        return self

    def encode(self, *args, **kwargs):
        return self._spm.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._spm.decode(*args, **kwargs)

    @property
    def unk_token_id(self):
        return self._spm.unk_id()

    @property
    def sep_token_id(self):
        return self._get_id(self.sep_token)

    @property
    def cls_token_id(self):
        return self._get_id(self.cls_token)

    @property
    def pad_token_id(self):
        return self._get_id(self.pad_token)

    def _get_id(self, piece):
        id_ = self._spm.piece_to_id(piece)
        if id_ == self._spm.unk_id():
            raise Exception(f"{piece} is not defined")
        return id_

    def __len__(self):
        return self._spm.get_piece_size()
