from gptchat.lib import load_config
from gptchat.lib import set_seed
from gptchat.lib import Request
from gptchat.lib import build_api
from .lib import generate
import transformers
import uvicorn


class Handler:
    def __init__(self, model, tokenizer, top_k, top_p):
        self._model = model
        self._tokenizer = tokenizer
        self._top_k = top_k
        self._top_p = top_p

    def generate(self, req: Request):
        response = generate(
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=self._top_k,
            top_p=self._top_p,
            max_length=30,
            text=req.context
        )
        return {
            "context": req.context,
            "response": response
        }


def main(config, host=None, port=None):
    params = load_config(config)
    print(params)
    set_seed(params.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(params.output.tokenizer_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)
    handler = Handler(
        model=model,
        tokenizer=tokenizer,
        top_p=params.top_p,
        top_k=params.top_k
    )

    app = build_api(handler)
    uvicorn.run(app=app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
