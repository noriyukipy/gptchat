from gptchat.lib import load_yaml
from gptchat.lib import set_seed
from gptchat.api import Request
from gptchat.api import Response
from gptchat.api import ModelInfo
from gptchat.api import build_api
from gptchat.tokenizers import TokenizerWrapper
from .config import Config
from tokenizers import Tokenizer
import tensorflow as tf
import transformers
import uvicorn


class Handler:
    def __init__(self, model, tokenizer, params):
        self._model = model
        self._tokenizer = tokenizer
        self._params = params

    def generate(self, req: Request):
        assert not req.response

        bad_words_ids = [
            self._tokenizer.encode(word, add_special_tokens=False)
            for word in self._params.pred.bad_words
        ]
        input_ids = self._tokenizer.encode(
            req.context, add_special_tokens=False,
        )
        output = self._model.generate(
            input_ids=tf.convert_to_tensor([input_ids]),
            do_sample=self._params.pred.do_sample,
            top_k=self._params.pred.top_k,
            top_p=self._params.pred.top_p,
            bad_words_ids=bad_words_ids,
            max_length=self._params.pred.max_length,
        )
        response = self._tokenizer.decode(output[0])

        # Clean up response
        cleaned_response = response.replace(" ", "")

        return Response(
            request=req,
            response=cleaned_response,
            model_info=ModelInfo(output=response),
        )


def main(config, host=None, port=None):
    params = Config(**load_yaml(config))
    print(params)
    set_seed(params.pred.seed)

    tokenizer = Tokenizer.from_file(params.output.tokenizer_file)
    tokenizer = TokenizerWrapper(tokenizer)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)
    handler = Handler(model, tokenizer, params)

    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
