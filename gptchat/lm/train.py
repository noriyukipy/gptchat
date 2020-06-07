from gptchat.lib import set_seed
from gptchat.lib import load_config
from gptchat.lm.lib import train
import transformers
import numpy as np
import os


def load_dataset(path):
    texts = []
    for line in open(path):
        texts.append(line.strip("\n"))
    return texts


def build_tokenizer(tokenizer_model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)
    return tokenizer


def build_data(tokenizer, texts, block_size):
    ids = []
    for text in texts:
        # Set add_special_tokens=False
        # not to add additional special tokens.
        tokens = tokenizer.tokenize(text)
        ids.extend(tokenizer.convert_tokens_to_ids(tokens))

    inputs = []
    labels = []
    for idx in range(0, len(ids)-block_size+1, block_size):
        sample = ids[idx:idx+block_size]
        inputs.append(sample[:-1])
        labels.append(sample[1:])
    return {"input_ids": np.array(inputs)}, np.array(labels)


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
    _params = load_config(config)
    print(_params)

    set_seed(_params.seed)

    _train_texts = load_dataset(_params.input.train_file)
    _valid_texts = load_dataset(_params.input.valid_file)

    # Build and save tokenizer
    tokenizer = build_tokenizer(_params.tokenizer_model_name)
    os.makedirs(_params.output.tokenizer_dir, exist_ok=True)
    # To be able to use AutoTokenizer when loading afterward,
    # the corresponded AutoConfig should be saved.
    # This is because the tokenizer is for BERT, which is
    # different from our actual model GPT2.
    # See more details about this issue here
    #   https://github.com/huggingface/transformers/issues/4197
    transformers.AutoConfig.from_pretrained(_params.tokenizer_model_name).save_pretrained(_params.output.tokenizer_dir)
    tokenizer.save_pretrained(_params.output.tokenizer_dir)

    # Build data
    _x_train, _y_train = build_data(tokenizer, _train_texts, _params.block_size)
    _x_valid, _y_valid = build_data(tokenizer, _valid_texts, _params.block_size)
    _x_test, _y_test = build_data(tokenizer, _valid_texts, _params.block_size)

    # Train model
    model = build_model(tokenizer, _params)
    _val_best_model = train(_params, model, tokenizer, _x_train, _y_train, _x_valid, _y_valid)
    _val_best_model.summary()

    # Evaluate best model with validation set
    _val_best_model.evaluate(_x_valid, _y_valid)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
