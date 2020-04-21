import responder
import torch
from collections import namedtuple
from transformers import BertJapaneseTokenizer
from transformers import GPT2DoubleHeadsModel
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
        response, candidates = self._predictor.predict(context=param.context)
        resp.media = {"context": param.context, "response": response, "candidates": candidates}

    def parse_param(self, req_dict):
        try:
            context = req_dict["context"]
        except KeyError:
            raise KeyError

        ReqParam = namedtuple("ReqParam", ["context"])
        return ReqParam(context=context)


class ResponsePredictor:
    def __init__(self, model, tokenizer, max_len, top_p, top_k, num_cands):
        self._model = model
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._top_p = top_p
        self._top_k = top_k
        self._num_cands = num_cands

    def predict(self, context):
        SEP, BOS, EOS = self._tokenizer.additional_special_tokens
        SEP_id, BOS_id, EOS_id = self._tokenizer.additional_special_tokens_ids

        seq = [
            [BOS] + self._tokenizer.tokenize(context),
            [SEP]
        ]
        tokens = sum(seq, [])
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = (
            [0] * len(seq[0]) +
            [1] * len(seq[1])
        )

        generator = TopPKGenerator(
            model=self._model,
            top_p=self._top_p,
            top_k=self._top_k
        )

        # build inputs to model
        input_ids = torch.tensor([token_ids for _ in range(self._num_cands)])
        token_type_ids = torch.tensor([segment_ids for _ in range(self._num_cands)])

        # keep EOS token
        # - to check whether generation completed or not
        # - to calculate sentence probability
        eos_index = [float("infinity") for _ in range(self._num_cands)]

        for _ in range(self._max_len):
            next_id, _ = generator.step(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            # update inputs to models for next prediction
            input_ids = torch.cat([input_ids, next_id], dim=1)
            token_type_ids = torch.cat([token_type_ids, torch.tensor([[1] for _ in range(self._num_cands)])], dim=1)

            # Update EOS token index
            for sample_idx, idx in torch.nonzero(input_ids == EOS_id):
                idx = int(idx)
                if idx < eos_index[sample_idx]:
                    eos_index[sample_idx] = idx
            try:
                eos_index.index(float("infinity"))
            except ValueError:
                break

        # if eos index is infinity, set it to the last index
        eos_index = [
            input_ids.size()[1]-1 if x == float("infinity") else x
            for x in eos_index
        ]

        # calculate multi-choice probability
        _, model_output = generator.step(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=torch.tensor(eos_index),
        )
        _, mc_prob = model_output[:2]

        # create candidates to return
        cands = []
        for idx in range(self._num_cands):
            gen_text = self._tokenizer.decode(input_ids[idx][len(token_ids):eos_index[idx]])
            prob = float(mc_prob[idx])
            cands.append({"text": gen_text, "score": prob})

        candidates = list(sorted(cands, key=lambda x: x["score"], reverse=True))
        response = candidates[0]["text"]

        return response, candidates


def main(model_dir, address=None, port=None, max_len=100, top_p=0.95, top_k=50, num_cands=3):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
    model = GPT2DoubleHeadsModel.from_pretrained(model_dir)
    model.eval()
    with torch.set_grad_enabled(False):
        predictor = ResponsePredictor(
            model=model,
            tokenizer=tokenizer,
            max_len=max_len,
            top_p=top_p,
            top_k=top_k,
            num_cands=num_cands,
        )

        # set routing
        api = responder.API()
        api.add_route("/chat", ChatHandler(predictor=predictor).handle)

        api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
