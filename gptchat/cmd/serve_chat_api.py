import responder
import torch
from collections import namedtuple
from transformers import BertJapaneseTokenizer
from transformers import GPT2LMHeadModel
from gptchat.lib.generator import TopPKGenerator


class ChatHandler:
    def __init__(self, predictor):
        self._predictor = predictor

    async def handle(self, req, resp):
        req_dict = await req.media()
        try:
            param = self.parse_param(req_dict)
        except KeyError:
            resp.status_code = 400
            resp.media = {"error": "request json body should have 'context' key"}
            return
        generated = self._predictor.predict(context=param.context)
        resp.media = {"context": param.context, "response": generated}

    def parse_param(self, req_dict):
        try:
            context = req_dict["context"]
        except KeyError:
            raise KeyError

        ReqParam = namedtuple("ReqParam", ["context"])
        return ReqParam(context=context)


class ResponsePredictor:
    def __init__(self, model, tokenizer, max_len, top_p, top_k):
        self._model = model
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._top_p = top_p
        self._top_k = top_k

    def predict(self, context):
        SEP, BOS, EOS = self._tokenizer.additional_special_tokens

        seq = [
            [BOS] + self._tokenizer.tokenize(context),
            [SEP]
        ]
        tokens = sum(seq, [])
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        segments = (
            [0] * len(seq[0]) +
            [1] * len(seq[1])
        )
        segment_ids = self._tokenizer.convert_tokens_to_ids(segments)

        generator = TopPKGenerator(self._model, top_p=self._top_p, top_k=self._top_k)
        input_ids = torch.tensor([token_ids])
        token_type_ids = torch.tensor([segment_ids])
        for _ in range(self._max_len):
            next_id = generator.step(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            input_ids = torch.cat([input_ids, next_id], dim=1)
            token_type_ids = torch.cat([token_type_ids, torch.tensor([[1]])], dim=1)

            if list(next_id) == self._tokenizer.convert_tokens_to_ids([EOS]):
                break

        gen_text = self._tokenizer.decode([int(x) for x in input_ids[0]])
        return gen_text


def main(model_dir, address=None, port=None, max_len=100, top_p=0.95, top_k=50):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    predictor = ResponsePredictor(
        model=model,
        tokenizer=tokenizer,
        max_len=max_len,
        top_p=top_p,
        top_k=top_k,
    )

    # set routing
    api = responder.API()
    api.add_route("/chat", ChatHandler(predictor=predictor).handle)

    api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
