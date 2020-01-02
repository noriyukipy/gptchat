def evaluate(tokenizer, model, device, data_loader, batch_size, block_size):
    """Calculate the perplexity of the corpus given by args."""
    model.eval()
    eval_loss = 0.0
    eval_lm_loss = 0.0
    eval_mc_loss = 0.0
    num_batches = 0
    for batch_idx, batch in enumerate(data_loader):
        batch = utils.to_device(device, batch)
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

        eval_lm_loss += lm_loss.mean().item()
        eval_mc_loss += mc_loss.mean().item()
        eval_loss += loss.mean().item()
        num_batches += 1

    eval_lm_loss = eval_lm_loss / num_batches
    eval_mc_loss = eval_mc_loss / num_batches
    eval_loss = eval_loss / num_batches
    return eval_lm_loss, eval_mc_loss, eval_loss


def main():

    # build dataset for evaluation
    eval_data_loader = None
    if eval_corpus:
        _, eval_data_loader = build_data(
            tokenizer, eval_corpus, batch_size, block_size, shuffle
        )

                # [TODO] Evaluation code here
                if eval_data_loader:
                    eval_lm_loss, eval_mc_loss, eval_loss = evaluate(
                        tokenizer,
                        model,
                        device,
                        eval_data_loader,
                        batch_size,
                        block_size
                    )
                    checkpoint_log += (
                        f", eval_lm_loss={eval_lm_loss:.4f}"
                        f", eval_mc_loss={eval_mc_loss:.4f}"
                        f", eval_loss={eval_loss:.4f}"
                    )
                    model_dirname += f"-loss_{eval_loss:.4f}"