from gptchat.lib import train_utils
from gptchat.lib.gpt_dataset import build_pretrained_data
import torch
import os
import tqdm
import logging
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def main(
    output_dir, data,
    tokenizer_model="gpt2",
    # model options
    n_embd=768, n_layer=12, n_head=12, n_ctx=1024,
    # train options
    num_epochs=1, batch_size=2, block_size=1024,
    seed=None, shuffle=True,
    # optimizer options
    lr=5e-5, eps=1e-08, max_grad_norm=1.0,
    # save options
    checkpoint_steps=10,
    # gpu
    gpu=False,
):
    if seed:
        train_utils.set_seed(seed)
    device = train_utils.get_device(use_gpu=gpu)

    # build tokenizer
    tokenizer = train_utils.build_tokenizer(model=tokenizer_model)
    logging.info(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # build dataset for training
    _, data_loader = build_pretrained_data(
        tokenizer=tokenizer,
        texts=[line.strip("\n") for line in open(data)],
        batch_size=batch_size,
        block_size=block_size,
        shuffle=shuffle,
    )

    # Define config
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_ctx=n_ctx,
        n_positions=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        num_labels=1,
    )

    # Check consistency between config and options
    assert config.n_ctx == config.n_positions == n_ctx == block_size

    # Initialize model
    model = GPT2LMHeadModel(config=config)
    model.to(device)

    # optimizer setup
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(data_loader)*num_epochs
    )

    # Save initial model before training
    train_utils.save(
        config, model, tokenizer,
        os.path.join(output_dir, "step_0")
    )

    # train
    total_loss = 0
    total_steps = 0
    for epoch_idx in range(num_epochs):
        num_batches = len(data_loader)
        epoch_iter = tqdm.tqdm(data_loader, desc="Iteration")
        for batch_idx, batch in enumerate(epoch_iter):
            (batch, ) = train_utils.to_device(device, [batch])

            # Model mode should be changed for each batching
            # becaause evaluation  may be done after training step.
            model.train()
            optimizer.zero_grad()
            # as mentioned in transformers document,
            # set labels == input_ids because the model
            # inside shifted the labels
            model_out = model(batch, labels=batch)
            loss = model_out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_steps += 1

            batch_log = (
                f"step {total_steps}, "
                f"epoch {epoch_idx+1}/{num_epochs}, "
                f"batch {batch_idx+1}/{num_batches}, "
                f"lr {scheduler.get_lr()[0]:.4e}, "
                f"loss {loss.item():.4f}"
            )

            # Checkpoint
            if total_steps % checkpoint_steps == 0:
                # print log
                checkpoint_log = "checkpoint " + batch_log
                print(checkpoint_log)
                # save model
                model_dirname = (
                    f"step_{total_steps}"
                    f"-epoch_{epoch_idx+1}"
                    f"-batch_{batch_idx+1}"
                )
                model_path = os.path.join(output_dir, model_dirname)
                train_utils.save(config, model, tokenizer, model_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
