from gptchat.lib import load_config
from gptchat.lib import set_seed
from .lib import generate
from fastapi import FastAPI
from pydantic import BaseModel
import transformers
import uvicorn


class Request(BaseModel):
    context: str
    response: str = None


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
        return {"response": response}


def build_api(handler):
    app = FastAPI(
        title="GPTChat",
        description="",
        version="0.0.0",

    )
    app.add_api_route("/generate", handler.generate, methods=["POST"])
    return app


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
