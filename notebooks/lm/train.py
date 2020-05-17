import attrdict
import transformers
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import yaml


def set_seed(seed):
    import numpy as np
    import tensorflow as tf
    import random
    import os

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


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


# Read https://github.com/huggingface/transformers/issues/2169
# to know more about how to train TFGPT2LMHead


class WarmupScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, learning_rate):
        super().__init__()

        self._warmup_steps = warmup_steps
        self._learning_rate = learning_rate

        # The argument passed to on_train_batch_begin
        # is resetted every epoch.
        # self._total_steps is used to keep total step
        self._total_steps = 0

    def on_train_batch_begin(self, step, logs=None):
        self._total_steps += 1
        step = self._total_steps

        if step > self._warmup_steps:
            return

        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self._learning_rate * (step / self._warmup_steps)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        #print('\nStep {}: lr is schedulerd {:.4e} -> {:.4e}]'.format(step, lr, float(tf.keras.backend.get_value(self.model.optimizer.lr))))

    
def train(params, tokenizer, x_train, y_train, x_valid, y_valid):
    # Prepare model directory and path
    model_save_dir = os.path.join(params.output_dir, "model")
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    checkpoint_model_path = os.path.join(params.output_dir, "ckpt.h5")
    tensorboard_output_dir = os.path.join(params.output_dir, "tensorboard")

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
            filepath=checkpoint_model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_output_dir,
            update_freq="batch",
            # To automatically refresh Tensorboard , set profile_batch=0
            # See more details here https://github.com/tensorflow/tensorboard/issues/2412
            profile_batch=0,  
        ),
        WarmupScheduler(total_steps * params.warmup_rate, params.learning_rate),
    ]
    
    # Train model
    tokenizer.save_pretrained(model_save_dir)   
    _ = model.fit(
        {"input_ids": x_train},
        y_train,
        epochs=params.num_epochs,
        batch_size=params.batch_size,
        callbacks=callbacks_list,
        validation_data=({"input_ids": x_valid}, y_valid),
    )

    # Restore the best model and save it as pretrained model format
    model.load_weights(checkpoint_model_path)
    model.save_pretrained(model_save_dir)

    # Save model with best performance
    return model


def main(config):
    _params = attrdict.AttrDict(yaml.load(open(config)))
    print(_params)

    set_seed(_params.seed)

    _train_texts = load_dataset(_params.data_dir + "/train.txt")
    _valid_texts = load_dataset(_params.data_dir + "/valid.txt")
    _test_texts = load_dataset(_params.data_dir + "/test.txt")

    _tokenizer = build_tokenizer(_params.tokenizer_model_name)
    
    _x_train, _y_train = build_data(_tokenizer, _train_texts, _params.block_size)
    _x_valid, _y_valid = build_data(_tokenizer, _valid_texts, _params.block_size)
    _x_test, _y_test = build_data(_tokenizer, _valid_texts, _params.block_size)

    # Train model
    _val_best_model = train(_params, _tokenizer, _x_train, _y_train, _x_valid, _y_valid)
    _val_best_model.summary()

    # Evaluate best model with validation set
    _val_best_model.evaluate(_x_valid, _y_valid)

    # Evaluate best model with test set
    _val_best_model.evaluate(_x_test, _y_test)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
