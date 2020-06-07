from gptchat.lib import set_seed
from gptchat.lib import load_config
from gptchat.lm.lib import train
import transformers
import numpy as np
import os
import collections


Sample = collections.namedtuple("Sample", ["context", "response"])


def load_dataset(path):
    samples = []
    for line in open(path):
        text = line.strip("\n")
        context, response = text.split("\t")
        samples.append(Sample(context=context, response=response))
    return samples


def encode_plus(tokenizer, context,
                response=None, add_sep_token=True, add_eos_token=True,
                pad_to_max_length=False, max_length=None):
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if add_sep_token:
        response_ids = [tokenizer.sep_token_id]
    if response:
        response_ids += tokenizer.encode(response, add_special_tokens=False)
    if add_eos_token:
        response_ids += [tokenizer.cls_token_id]

    input_ids = context_ids + response_ids
    token_type_ids = [0] * len(context_ids) + [1] * len(response_ids)
    attention_mask = [1] * (len(context_ids) + len(response_ids))

    if max_length:
        input_ids = input_ids[:max_length]
        token_type_ids = token_type_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    if len(input_ids) < max_length:
        diff_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * diff_len
        token_type_ids += [tokenizer.pad_token_id] * diff_len
        attention_mask += [0] * diff_len

    tensor = {
        "input_ids":      np.array(input_ids),
        "token_type_ids": np.array(token_type_ids),
        "attention_mask": np.array(attention_mask),
    }

    return tensor


def build_data(tokenizer, samples, max_length):
    input_ = {
        "input_ids":      [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    labels = []
    for sample in samples:
        tensor = encode_plus(
            tokenizer, sample.context, sample.response,
            pad_to_max_length=True, max_length=max_length
        )
        for key in input_:
            input_[key].append(tensor[key][:-1])
        labels.append(tensor["input_ids"][1:])

    input_ = {key: np.array(val) for key, val in input_.items()}
    labels = np.array(labels)

    return input_, labels


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
    tokenizer = transformers.AutoTokenizer.from_pretrained(params.input.pretrained_tokenizer_dir)
    os.makedirs(params.output.tokenizer_dir, exist_ok=True)
    # To be able to use AutoTokenizer when loading afterward,
    # the corresponded AutoConfig should be saved.
    # This is because the tokenizer is for BERT, which is
    # different from our actual model GPT2.
    # See more details about this issue here
    #   https://github.com/huggingface/transformers/issues/4197
    transformers.AutoConfig.from_pretrained(params.input.pretrained_tokenizer_dir).save_pretrained(params.output.tokenizer_dir)
    tokenizer.save_pretrained(params.output.tokenizer_dir)

    # Build data
    x_train, y_train = build_data(tokenizer, train_texts, params.max_length)
    x_valid, y_valid = build_data(tokenizer, valid_texts, params.max_length)

    # Train model
    model = transformers.TFGPT2LMHeadModel.from_pretrained(params.input.pretrained_model_dir)
    val_best_model = train(params, model, tokenizer, x_train, y_train, x_valid, y_valid)
    val_best_model.summary()

    # Evaluate best model with validation set
    val_best_model.evaluate(x_valid, y_valid)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
