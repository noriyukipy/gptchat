from .callback import WarmupScheduler
from .callback import TransformersCheckpoint
import transformers
import tensorflow.keras as keras
import tensorflow as tf
import os


def init_model(vocab_size, params):
    config = transformers.GPT2Config(
        vocab_size=vocab_size,
        n_ctx=params.n_ctx,
        n_positions=params.n_ctx,
        n_embd=params.n_embd,
        n_layer=params.n_layer,
        n_head=params.n_head,
    )
    model = transformers.TFGPT2LMHeadModel(config=config)
    return model


def load_or_init_model(pretrained_model_dir, vocab_size, params):
    # Train model
    if pretrained_model_dir:
        print(f"Load model from {pretrained_model_dir}")
        model = transformers.TFGPT2LMHeadModel.from_pretrained(pretrained_model_dir)
    else:
        print(f"Initialize model with parameters: {params}")
        model = init_model(vocab_size, params)

    return model


def cross_entropy_loss_with_padding(num_labels, pad_token_id):
    loss_fct = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=keras.losses.Reduction.NONE
    )

    def loss(y_true, y_pred):
        input_mask = tf.not_equal(y_true, pad_token_id)
        active_loss = tf.reshape(input_mask, (-1,))
        logits = tf.reshape(y_pred, (-1, num_labels))
        active_logits = tf.boolean_mask(logits, active_loss)

        train_labels = tf.reshape(y_true, (-1,))
        active_labels = tf.boolean_mask(train_labels, active_loss)
        cross_entropy = loss_fct(active_labels, active_logits)

        return cross_entropy

    return loss


# To know more about how to train TFGPT2LMHead, read
#   https://github.com/huggingface/transformers/issues/2169
def train(params, model, tokenizer, train_dataset, valid_dataset):
    # Prepare model directory and path
    os.makedirs(params.output.model_dir, exist_ok=True)

    # Compile model
    # Set from_logits=True because TFGPT2LMHeadModel returns the logits (before Softmax)
    # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cross_entropy_loss_with_padding(
        num_labels=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
    )

    # Create optimizer
    total_steps = len(train_dataset) * params.train.num_epochs
    optimizer = keras.optimizers.Adam(
        lr=params.train.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,  # default is 1e-07
        clipnorm=params.train.max_grad_norm,  # cilipping gradient by L2 norm
    )

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        # metrics=[
        #     keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        #     keras.metrics.SparseCategoricalAccuracy(),
        # ],
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params.train.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=params.output.checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        TransformersCheckpoint(model=model, save_dir=params.output.model_dir),
        keras.callbacks.TensorBoard(
            log_dir=params.output.tensorboard_dir,
            update_freq="batch",
            # To automatically refresh Tensorboard , set profile_batch=0
            # See more details here https://github.com/tensorflow/tensorboard/issues/2412
            profile_batch=0,
        ),
        WarmupScheduler(
            total_steps * params.train.warmup_rate, params.train.learning_rate
        ),
    ]

    # Train model
    model.fit(
        train_dataset,
        epochs=params.train.num_epochs,
        callbacks=callbacks_list,
        validation_data=valid_dataset,
    )

    # Restore the best model and save it as pretrained model format
    # If restore_best_weights=False, this process is required
    model.load_weights(params.output.checkpoint_path)
    model.save_pretrained(params.output.model_dir)

    # Save model with best performance
    return model
