import responder
import torch
from transformers import BertJapaneseTokenizer
from transformers import GPT2LMHeadModel
from gptchat.lib.generator import TopPKGenerator


class LMGenerator:
    def __init__(self, model, tokenizer, max_len, top_p, top_k):
        self._model = model
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._top_p = top_p
        self._top_k = top_k

    async def generate(self, req, resp):
        # Validate input
        req_json = await req.media()
        print(req_json)
        try:
            text = req_json["text"]
        except KeyError:
            resp.status_code = 400
            resp.media = {"error": "request json body should have 'text' key"}
            return

        # Generate text
        gen_text = self._generate_text(text)

        # Set response
        resp.media = {"text": gen_text}

    def _generate_text(self, text):
        BOS_id, EOS_id = self._tokenizer.additional_special_tokens_ids

        generator = TopPKGenerator(
            model=self._model,
            top_p=self._top_p,
            top_k=self._top_k,
            bad_ids=[0, 1],  # Ignore [PAD] and [UNK]
        )

        ids = self._tokenizer.encode(text, add_special_tokens=False)
        ids = [BOS_id] + ids
        input_ids = torch.tensor([ids])
        for _ in range(self._max_len):
            next_id, _ = generator.step(input_ids=input_ids)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id == EOS_id:
                break

        gen_text = self._tokenizer.decode([int(x) for x in input_ids[0]])
        return gen_text


def build_api(model_dir, max_len, top_p, top_k):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    gen = LMGenerator(model, tokenizer, max_len, top_p, top_k)

    api = responder.API()
    api.add_route("/generate", gen.generate)
    return api


def main(model_dir, address=None, port=None, max_len=100, top_p=0.95, top_k=50):
    api = build_api(model_dir, max_len, top_p, top_k)
    api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
