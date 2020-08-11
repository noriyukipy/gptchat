from gptchat.lib import load_config
from gptchat.lib import set_seed
from .lib import generate
from .lib import generate_prepare_inputs_for_generation
import transformers
import types


def main(config):
    params = load_config(config)
    print(params)
    set_seed(params.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(params.output.tokenizer_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)
    # Replace generation initializer
    method = types.MethodType(
        generate_prepare_inputs_for_generation(sep_token_id=tokenizer.sep_token_id),
        model,
    )
    model.prepare_inputs_for_generation = method

    bad_words_ids = [
        tokenizer.encode(word) for word in params.bad_words
    ]

    context = "おはよう"
    response = ""

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
