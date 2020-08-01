from gptchat.lib import load_config
from gptchat.lib import set_seed
from gptchat.lib import Request
from gptchat.lib import Response
from gptchat.lib import ModelInfo
from gptchat.lib import build_api
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
            for word in self._params.bad_words
        ]
        input_ids = self._tokenizer.encode(
            req.context, add_special_tokens=False, return_tensors="tf"
        )
        output = self._model.generate(
            input_ids=input_ids,
            do_sample=self._params.do_sample,
            top_k=self._params.top_k,
            top_p=self._params.top_p,
            bad_words_ids=bad_words_ids,
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
    params = load_config(config)
    print(params)
    set_seed(params.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(params.output.tokenizer_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)
    handler = Handler(model, tokenizer, params)

    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
