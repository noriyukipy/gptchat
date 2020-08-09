import sentencepiece as spm
from pydantic import BaseModel
from typing import List
from gptchat.lib import load_yaml


class TrainConfig(BaseModel):
    input: str
    model_prefix: str
    vocab_size: int
    user_defined_symbols: List[str]


class Config(BaseModel):
    train: TrainConfig


def main(config):
    # Parse config file and convert to object
    config = Config(**load_yaml(config))

    # Then train it!
    spm.SentencePieceTrainer.train(
        input=config.train.input,
        model_prefix=config.train.model_prefix,
        vocab_size=config.train.vocab_size,
        user_defined_symbols=config.train.user_defined_symbols
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
