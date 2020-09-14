from gptchat.lib import set_seed
from gptchat.lib import load_yaml
from gptchat.gpt2 import load_or_init_model
from gptchat.gpt2 import train
from gptchat.tokenizers import SentencePieceTokenizer
from .config import Config
import numpy as np
import tensorflow as tf
import math


def load_dataset(path):
    contexts = []
    for line in open(path):
        context = line.strip("\n").split("\t")
        contexts.append(context)
    return contexts


def encode_plus(
    tokenizer,
    context,
    pad_to_max_length=False,
    max_length=None,
):
    """
    Args:
        context (List[str]):
    """
    context_ids_list = tokenizer.encode(context)
    input_ids = [tokenizer.sep_token_id]
    for cids in context_ids_list:
        input_ids = input_ids + cids + [tokenizer.sep_token_id]

    attention_mask = [1] * len(input_ids)

    if max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    if pad_to_max_length and len(input_ids) < max_length:
        diff_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * diff_len
        attention_mask += [0] * diff_len

    tensor = {
        "input_ids": np.array(input_ids),
        "attention_mask": np.array(attention_mask),
    }

    return tensor


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, tokenizer, contexts, max_length, batch_size):
        self._tokenizer = tokenizer
        self._contexts = contexts
        self._max_length = max_length
        self._batch_size = batch_size

    def __getitem__(self, idx):
        contexts = [
            self._contexts[i]
            for i in range(
                idx * self._batch_size,
                min((idx + 1) * self._batch_size, len(self._contexts)),
            )
        ]
        return build_data(self._tokenizer, contexts, self._max_length)

    def __len__(self):
        return math.ceil(len(self._contexts) / self._batch_size)


def build_data(tokenizer, contexts, max_length):
    input_ = {
        "input_ids": [],
        "attention_mask": [],
    }
    labels = []
    for context in contexts:
        tensor = encode_plus(tokenizer, context, pad_to_max_length=True, max_length=max_length)
        for key in input_:
            input_[key].append(tensor[key][:-1])
        labels.append(tensor["input_ids"][1:])

    input_ = {key: np.array(val) for key, val in input_.items()}
    labels = np.array(labels)

    return input_, labels


def main(config):
    params = Config(**load_yaml(config))
    print(params)

    # Set seed
    set_seed(params.train.seed)

    train_texts = load_dataset(params.input.train_file)
    valid_texts = load_dataset(params.input.valid_file)

    # Build and save tokenizer
    tokenizer = SentencePieceTokenizer().load(params.input.tokenizer_file)

    # Build dataset
    train_dataset = Dataset(
        tokenizer, train_texts, params.train.max_length, params.train.batch_size
    )
    valid_dataset = Dataset(
        tokenizer, valid_texts, params.train.max_length, params.train.batch_size
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
