from gptchat.lib import set_seed
from gptchat.lib import load_yaml
from gptchat.gpt2 import train
from gptchat.gpt2 import load_or_init_model
from .config import Config
from gptchat.tokenizers import TokenizerWrapper
from tokenizers import Tokenizer
import tensorflow as tf
import transformers
import numpy as np
import math


def load_dataset(path):
    texts = []
    for line in open(path):
        texts.append(line.strip("\n"))
    return texts


def build_tokenizer(tokenizer_model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)
    return tokenizer


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, tokenizer, texts, block_size, batch_size):
        ids = []
        for text in texts:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))

        samples = []
        for idx in range(0, len(ids) - block_size + 1, block_size):
            sample = ids[idx : idx + block_size]
            samples.append(sample)

        # Define attributes
        self._batch_size = batch_size
        self._samples = samples

    def __getitem__(self, idx):
        inputs = []
        labels = []

        for i in range(
            idx * self._batch_size,
            min((idx + 1) * self._batch_size, len(self._samples)),
        ):
            sample = self._samples[i]
            inputs.append(sample[:-1])
            labels.append(sample[1:])

        return {"input_ids": np.array(inputs)}, np.array(labels)

    def __len__(self):
        return math.ceil(len(self._samples) / self._batch_size)


def main(config):
    params = Config(**load_yaml(config))
    print(params)

    set_seed(params.train.seed)

    train_texts = load_dataset(params.input.train_file)
    valid_texts = load_dataset(params.input.valid_file)

    # Build and save tokenizer
    # Build and save tokenizer
    tokenizer = Tokenizer.from_file(params.input.tokenizer_file)
    tokenizer.save(params.output.tokenizer_file)
    tokenizer = TokenizerWrapper(tokenizer)

    # Build data
    train_dataset = Dataset(
        tokenizer, train_texts, params.train.block_size, params.train.batch_size
    )
    valid_dataset = Dataset(
        tokenizer, valid_texts, params.train.block_size, params.train.batch_size
    )

    # Train model
    model = load_or_init_model(
        pretrained_model_dir=params.input.pretrained_model_dir,
        vocab_size=len(tokenizer),
        params=params.model_params,
    )
    val_best_model = train(params, model, tokenizer, train_dataset, valid_dataset)
    val_best_model.summary()

    # Evaluate best model with validation set
    val_best_model.evaluate(valid_dataset)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
