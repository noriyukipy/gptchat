from gptchat.lib import set_seed
from gptchat.lib import load_config
from gptchat.lm.lib import train
import tensorflow as tf
import transformers
import numpy as np
import os
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
            # Set add_special_tokens=False
            # not to add additional special tokens.
            tokens = tokenizer.tokenize(text)
            ids.extend(tokenizer.convert_tokens_to_ids(tokens))

        samples = []
        for idx in range(0, len(ids)-block_size+1, block_size):
            sample = ids[idx:idx+block_size]
            samples.append(sample)

        # Define attributes
        self._batch_size = batch_size
        self._samples = samples

    def __getitem__(self, idx):
        inputs = []
        labels = []

        for i in range(idx*self._batch_size, min((idx+1)*self._batch_size, len(self._samples))):
            sample = self._samples[i]
            inputs.append(sample[:-1])
            labels.append(sample[1:])

        return {"input_ids": np.array(inputs)}, np.array(labels)

    def __len__(self):
        return math.ceil(len(self._samples) / self._batch_size)


def build_model(tokenizer, params):
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer),
        n_ctx=params.n_ctx,
        n_positions=params.block_size,
        n_embd=params.n_embd,
        n_layer=params.n_layer,
        n_head=params.n_head,
    )
    model = transformers.TFGPT2LMHeadModel(config=config)
    return model


def main(config):
    params = load_config(config)
    print(params)

    set_seed(params.seed)

    train_texts = load_dataset(params.input.train_file)
    valid_texts = load_dataset(params.input.valid_file)

    # Build and save tokenizer
    tokenizer = build_tokenizer(params.tokenizer_model_name)
    os.makedirs(params.output.tokenizer_dir, exist_ok=True)
    # To be able to use AutoTokenizer when loading afterward,
    # the corresponded AutoConfig should be saved.
    # This is because the tokenizer is for BERT, which is
    # different from our actual model GPT2.
    # See more details about this issue here
    #   https://github.com/huggingface/transformers/issues/4197
    transformers.AutoConfig.from_pretrained(params.tokenizer_model_name).save_pretrained(params.output.tokenizer_dir)
    tokenizer.save_pretrained(params.output.tokenizer_dir)

    # Build data
    train_dataset = Dataset(tokenizer, train_texts, params.block_size, params.batch_size)
    valid_dataset = Dataset(tokenizer, valid_texts, params.block_size, params.batch_size)

    # Train model
    model = build_model(tokenizer, params)
    val_best_model = train(params, model, tokenizer, train_dataset, valid_dataset)
    val_best_model.summary()

    # Evaluate best model with validation set
    val_best_model.evaluate(valid_dataset)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
