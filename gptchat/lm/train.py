from gptchat.lib import set_seed
from gptchat.lib import WarmupScheduler
from gptchat.lib import load_config
import transformers
import numpy as np
import tensorflow.keras as keras
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
    return np.array(inputs), np.array(labels)


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


# To know more about how to train TFGPT2LMHead, read
#   https://github.com/huggingface/transformers/issues/2169
def train(params, tokenizer, x_train, y_train, x_valid, y_valid):
    # Prepare model directory and path
    os.makedirs(params.output.model_dir, exist_ok=True)

    # Compile model
    # Set from_logits=True because TFGPT2LMHeadModel returns the logits (before Softmax)
    model = build_model(tokenizer, params)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Create optimizer
    total_steps = int(len(x_train) / params.batch_size) * params.num_epochs
    optimizer = keras.optimizers.Adam(
        lr=params.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,  # default is 1e-07
        clipnorm=params.max_grad_norm  # cilipping gradient by L2 norm
    )

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        metrics=[
            keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
            keras.metrics.SparseCategoricalAccuracy(),
        ],
    )
    
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params.patience,
            # EarlyStopping callback does keep the previous epoch model even if the performance gets worse.
            # To restore the best model, load weights from checkpoint which keeps the best only.
            restore_best_weights=False
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=params.output.checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=params.output.tensorboard_dir,
            update_freq="batch",
            # To automatically refresh Tensorboard , set profile_batch=0
            # See more details here https://github.com/tensorflow/tensorboard/issues/2412
            profile_batch=0,  
        ),
        WarmupScheduler(total_steps * params.warmup_rate, params.learning_rate),
    ]
    
    # Train model
    model.fit(
        {"input_ids": x_train},
        y_train,
        epochs=params.num_epochs,
        batch_size=params.batch_size,
        callbacks=callbacks_list,
        validation_data=({"input_ids": x_valid}, y_valid),
    )

    # Restore the best model and save it as pretrained model format
    model.load_weights(params.output.checkpoint_path)
    model.save_pretrained(params.output.model_dir)

    # Save model with best performance
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
    _val_best_model = train(_params, tokenizer, _x_train, _y_train, _x_valid, _y_valid)
    _val_best_model.summary()

    # Evaluate best model with validation set
    _val_best_model.evaluate(_x_valid, _y_valid)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
