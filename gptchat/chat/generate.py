from gptchat.lib import train_utils
from gptchat.lib import chat_utils
from transformers import GPT2DoubleHeadsModel


def main(model, max_len=30, top_p=0.9):
    # Load tokenizer and model
    tokenizer = train_utils.build_tokenizer(model)
    model = GPT2DoubleHeadsModel.from_pretrained(model)

    while True:
        text = input(">>> ")
        gen = chat_utils.generate(
            tokenizer=tokenizer,
            model=model,
            text=text,
            max_len=max_len,
            top_p=top_p
        )
        print(gen)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
