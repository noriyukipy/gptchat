import torch
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from gptchat.lib import train_utils
from gptchat.lib.gpt_dataset import build_pretrained_data


def evaluate(tokenizer, model, device, data_loader):
    """Calculate the perplexity of the corpus given by args."""
    model.eval()
    eval_loss = 0.0
    num_batches = 0
    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to(device)
        model_out = model(batch, labels=batch)
        loss = model_out[0]
        eval_loss += loss.mean().item()
        num_batches += 1

    eval_loss = eval_loss / num_batches
    ppl = torch.exp(torch.tensor(eval_loss))
    return ppl


def main(model, data, batch_size=2, gpu=False):
    device = train_utils.get_device(use_gpu=gpu)

    # build tokenizer
    tokenizer = train_utils.build_tokenizer(model)
    config = GPT2Config.from_pretrained(model)
    model = GPT2LMHeadModel.from_pretrained(model)
    model.to(device)

    # build dataset for training
    _, data_loader = build_pretrained_data(
        tokenizer=tokenizer,
        texts=[line.strip("\n") for line in open(data)],
        batch_size=batch_size,
        block_size=config.n_ctx,
        shuffle=False
    )
    ppl = evaluate(
        tokenizer,
        model,
        device,
        data_loader,
    )
    print(f"Perplexity {ppl:.4f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
