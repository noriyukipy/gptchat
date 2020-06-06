from gptchat.lib import load_config
import transformers
import os
from train import set_seed
import responder


class APIHandler:
    def __init__(self, generator):
        self._generator = generator

    async def generate(self, req, resp):
        # Validate input
        req_json = await req.media()
        try:
            context = req_json["context"]
            response = req_json.get("response", "")
        except KeyError:
            resp.status_code = 400
            resp.media = {"error": "request json body should have 'context' key"}
            return

        # Generate text
        gen_text = self._generator.generate(context, response)

        # Set response
        resp.media = {
            "request": req_json,
            "response": gen_text,
        }


class LMGenerator:
    def __init__(self, model, tokenizer, params):
        self._model = model
        self._tokenizer = tokenizer
        self._params = params
    
    def generate(self, context, response):
        print(response)
        assert response == ""

        bad_words_ids = [self._tokenizer.encode(word, add_special_tokens=False) for word in self._params.bad_words]
        output = self._model.generate(
            input_ids=self._tokenizer.encode(context, add_special_tokens=False, return_tensors="tf"),
            do_sample=self._params.do_sample,
            top_k=self._params.top_k,
            top_p=self._params.top_p,
            bad_words_ids=bad_words_ids,
        )
        return self._tokenizer.decode(output[0])


def build_api(model_dir, params):
    tokenizer = transformers.AutoTokenizer.from_pretrained(params.output.tokenizer_dir)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(params.output.model_dir)

    generator = LMGenerator(model, tokenizer, params)
    handler = APIHandler(generator=generator)

    api = responder.API()
    api.add_route("/generate", handler.generate)
    return api


def main(config, address=None, port=None):
    params = load_config(config)
    print(params)
    set_seed(params.seed)

    model_dir = os.path.join(params.output_dir, "model")
    api = build_api(model_dir, params)
    api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
