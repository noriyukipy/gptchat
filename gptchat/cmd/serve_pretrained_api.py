import responder
import torch
from transformers import BertJapaneseTokenizer
from transformers import GPT2LMHeadModel
from gptchat.lib.generator import TopPGenerator


class LMGenerator:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    async def generate(self, req, resp):
        # Validate input
        req_json = await req.media()
        print(req_json)
        try:
            text = req_json["text"]
            max_len = req_json.get("max_len", 100)
            top_p = req_json.get("top_p", 0.8)
        except KeyError:
            resp.status_code = 400
            resp.media = {"error": "request json body should have 'text' key"}
            return

        # Generate text
        gen_text = self._generate_text(text, max_len, top_p)

        # Set response
        resp.media = {"text": gen_text}

    def _generate_text(self, text, max_len, top_p):
        generator = TopPGenerator(self._model, top_p)

        ids = self._tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([ids])
        for _ in range(max_len):
            next_id = generator.step(input_ids=input_ids)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        gen_text = self._tokenizer.decode([int(x) for x in input_ids[0]])
        return gen_text


def build_api(model_dir):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    gen = LMGenerator(model, tokenizer)

    api = responder.API()
    api.add_route("/generate", gen.generate)
    return api


def main(model_dir, address=None, port=None):
    api = build_api(model_dir)
    api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
