from gptchat.lib import load_config
from gptchat.lib import set_seed
from .lib import generate
import transformers


def main(config):
    params = load_config(config)
    print(params)
    set_seed(params.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(params.output.tokenizer_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)

    bad_words_ids = [
        tokenizer.encode(word, add_special_tokens=False)
        for word in params.bad_words
    ]

    context = "今日は疲れた"
    response = "明日"

    output = generate(
        model=model,
        tokenizer=tokenizer,
        top_k=params.top_k,
        top_p=params.top_p,
        max_length=30,
        context=context,
        response=response,
        bad_words_ids=bad_words_ids,
    )
    print(output)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
