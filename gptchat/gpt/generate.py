import torch
from transformers import GPT2LMHeadModel
from gptchat.lib.generator import TopPGenerator
from gptchat.lib import train_utils


def generate(tokenizer, model, text, max_len, top_p):
    """
    Returns:
        str: Generated text
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids])
    generator = TopPGenerator(model, top_p=top_p)

    for _ in range(max_len):
        next_id = generator.step(input_ids=input_ids)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode([int(x) for x in input_ids[0]])


def main(model, max_len=30, top_p=0.9):
    # Load tokenizer and model
    tokenizer = train_utils.build_tokenizer(model)
    model = GPT2LMHeadModel.from_pretrained(model)

    while True:
        text = input(">>> ")
        gen = generate(tokenizer, model, text, max_len, top_p=top_p)
        print(gen)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
