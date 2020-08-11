from pydantic import BaseModel
from typing import List, Union
from gptchat.lib import load_yaml
from gptchat.tokenizers import SentencePieceTokenizer


class TrainConfig(BaseModel):
    input: str
    model_prefix: str
    vocab_size: int
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    pad_piece: str
    unk_piece: str
    bos_piece: str
    eos_piece: str
    user_defined_symbols: List[str]
    input_sentence_size: Union[None, int]
    shuffle_input_sentence: bool


class Config(BaseModel):
    train: TrainConfig


def main(config):
    # Parse config file and convert to object
    config = Config(**load_yaml(config))
    print(config)

    # Then train it!
    tokenizer = SentencePieceTokenizer()
    tokenizer.train(**config.train.dict())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
