from gptchat.lib import WarmupScheduler
import tensorflow.keras as keras
import os


# To know more about how to train TFGPT2LMHead, read
#   https://github.com/huggingface/transformers/issues/2169
def train(params, model, tokenizer, x_train, y_train, x_valid, y_valid):
    # Prepare model directory and path
    os.makedirs(params.output.model_dir, exist_ok=True)

    # Compile model
    # Set from_logits=True because TFGPT2LMHeadModel returns the logits (before Softmax)
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

