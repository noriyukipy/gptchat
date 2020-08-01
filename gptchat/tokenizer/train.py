from tokenizers import SentencePieceBPETokenizer
from pydantic import BaseModel
from typing import List
from gptchat.lib import load_yaml


class TrainConfig(BaseModel):
    files: List[str]
    output_file: str
    add_prefix_space: bool
    unk_token: str
    vocab_size: int
    min_frequency: int
    special_tokens: List[str]
    limit_alphabet: int
    initial_alphabet: List[str]


class Config(BaseModel):
    train: TrainConfig


def main(config):
    # Parse config file and convert to object
    config = Config(**load_yaml(config))

    # Prepare tokenizer
    tokenizer = SentencePieceBPETokenizer(
        add_prefix_space=config.train.add_prefix_space,
        unk_token=config.train.unk_token,
    )

    # Then train it!
    tokenizer.train(
        files=config.train.files,
        vocab_size=config.train.vocab_size,
        min_frequency=config.train.min_frequency,
        special_tokens=config.train.special_tokens,
        limit_alphabet=config.train.limit_alphabet,
        initial_alphabet=config.train.initial_alphabet,
    )
    print(f"Vocab size={tokenizer.get_vocab_size()}")

    # Save vocabulary file

    output_file = config.train.output_file
    print(f"Save model at {output_file}")
    tokenizer.save(output_file)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
