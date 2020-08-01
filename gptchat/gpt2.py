import transformers


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
