from gptchat.lib import WarmupScheduler
import tensorflow.keras as keras
import tensorflow as tf
import os


def cross_entropy_loss_with_padding(num_labels, pad_token_id):
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
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
        num_labels=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )

    # Create optimizer
    total_steps = len(train_dataset) * params.num_epochs
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
        # metrics=[
        #     keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        #     keras.metrics.SparseCategoricalAccuracy(),
        # ],
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
        train_dataset,
        epochs=params.num_epochs,
        callbacks=callbacks_list,
        validation_data=valid_dataset,
    )

    # Restore the best model and save it as pretrained model format
    model.load_weights(params.output.checkpoint_path)
    model.save_pretrained(params.output.model_dir)

    # Save model with best performance
    return model

