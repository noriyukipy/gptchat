from gptchat.lib.chat_dataset import DialogDataset
from gptchat.lib.chat_dataset import PaddingCollation
from gptchat.lib import train_utils
from gptchat.lib import special_tokens
from transformers import GPT2DoubleHeadsModel
from transformers import GPT2Config
import logging
import torch
import os
import tqdm
from transformers import AdamW


def build_data(tokenizer, corpus, batch_size, num_distructors, shuffle):
    """Prepare DataSet and DataLoader."""
    inputs = []
    for line_num, line in enumerate(open(corpus)):
        text = line.strip("\n")
        items = text.split("\t")
        distructors = items[2:]
        if len(distructors) < num_distructors:
            print(
                f"The number of distructors is less than "
                f"{num_distructors} "
                f"at line {line_num}"
            )
        inputs.append((items[0], items[1], distructors[:num_distructors]))
    data_set = DialogDataset(tokenizer, inputs)
    padding_collation = PaddingCollation(padding_value=0)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=padding_collation.apply
    )
    return data_set, data_loader


def add_special_tokens(tokenizer, model):
    sp_tokens = special_tokens.get()
    num_added_tokens = tokenizer.add_tokens(sp_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(
        f"Added {num_added_tokens} special tokens to tokenizer"
        " and resize model token embedding."
    )


def main(
    model, output_dir, data,
    # train options
    num_distructors=1, num_epochs=1, batch_size=2, seed=None,
    shuffle=True,
    # optimizer option
    lr=5e-5, eps=1e-08, max_grad_norm=1.0,
    # save options
    checkpoint_steps=10,
    # gpu
    gpu=False,
):
    if seed:
        train_utils.set_seed(seed)
    device = train_utils.get_device(use_gpu=gpu)

    # Load tokenizer
    tokenizer = train_utils.build_tokenizer(model)
    logging.info(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # Load model
    config = GPT2Config.from_pretrained(model)
    model = GPT2DoubleHeadsModel.from_pretrained(model)
    model.to(device)

    # Add special tokens to tokenizer and expand embedding vocab size of model
    add_special_tokens(tokenizer, model)

    # build dataset for training
    _, data_loader = build_data(
        tokenizer, data, batch_size, num_distructors, shuffle
    )

    # optimizer setup
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    # Save initial model before training
    train_utils.save(
        config, model, tokenizer,
        os.path.join(output_dir, "step_0")
    )

    total_loss = 0
    total_steps = 0
    for epoch_idx in range(num_epochs):
        epoch_iter = tqdm.tqdm(data_loader, desc="Iteration")
        for batch_idx, batch in enumerate(epoch_iter):
            batch = train_utils.to_device(device, batch)

            # Model mode should be changed for each batching
            # becaause evaluation  may be done after training step.
            model.train()
            optimizer.zero_grad()
            # as mentioned in transformers document,
            # set labels == input_ids because the model
            # inside shifted the labels
            model_out = model(
                input_ids=batch[0],
                token_type_ids=batch[1],
                lm_labels=batch[2],
                mc_token_ids=batch[3],
                mc_labels=batch[4],
            )
            lm_loss, mc_loss, lm_score, mc_score = model_out[:4]

            lm_weight = 2.0
            mc_weight = 1.0
            loss = lm_loss * lm_weight + mc_loss * mc_weight

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

            batch_log = (
                f"epoch {epoch_idx}, "
                f"batch {batch_idx}, "
                f"lm_loss {lm_loss:.4f}, "
                f"mc_loss {mc_loss:.4f}, "
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
