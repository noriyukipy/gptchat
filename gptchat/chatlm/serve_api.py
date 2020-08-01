from gptchat.api import Request
from gptchat.api import Response
from gptchat.api import ModelInfo
from gptchat.api import build_api
from gptchat.lib import set_seed
from gptchat.lib import load_yaml
from gptchat.tokenizers import TokenizerWrapper
from .config import Config
from .lib import generate_prepare_inputs_for_generation
from .lib import generate
from tokenizers import Tokenizer
import types
import transformers
import uvicorn


class Handler:
    def __init__(self, model, tokenizer, top_k, top_p, max_length, bad_words_ids):
        self._model = model
        self._tokenizer = tokenizer
        self._top_k = top_k
        self._top_p = top_p
        self._max_length = max_length
        self._bad_words_ids = bad_words_ids

    def generate(self, req: Request):
        response, model_output = generate(
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=self._top_k,
            top_p=self._top_p,
            max_length=self._max_length,
            context=req.context,
            response=req.response,
            bad_words_ids=self._bad_words_ids,
        )
        return Response(
            request=req, response=response, model_info=ModelInfo(output=model_output)
        )


def main(config, host=None, port=None):
    params = Config(**load_yaml(config))
    print(params)
    set_seed(params.pred.seed)

    tokenizer = Tokenizer.from_file(params.output.tokenizer_file)
    tokenizer = TokenizerWrapper(tokenizer)

    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)
    # Replace generation initializer
    method = types.MethodType(
        generate_prepare_inputs_for_generation(sep_token_id=tokenizer.sep_token_id),
        model,
    )
    model.prepare_inputs_for_generation = method

    bad_words_ids = [
        tokenizer.encode(word, add_special_tokens=False)
        for word in params.pred.bad_words
    ]
    handler = Handler(
        model=model,
        tokenizer=tokenizer,
        top_k=params.pred.top_k,
        top_p=params.pred.top_p,
        max_length=params.pred.max_length,
        bad_words_ids=bad_words_ids,
    )

    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
