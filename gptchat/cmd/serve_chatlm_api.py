import responder
import torch
from transformers import BertJapaneseTokenizer
from transformers import GPT2LMHeadModel
from gptchat.lib.generator import TopPKGenerator
from gptchat.lib.response import split_context_response_ids
from gptchat.lib.response import filter_out_by_coverage_ratio
from gptchat.lib.chatlm import ChatLMModelInputBuilder


class APIHandler:
    def __init__(self, generator):
        self._generator = generator

    async def generate(self, req, resp):
        # Validate input
        req_json = await req.media()
        print(req_json)
        try:
            context = req_json["context"]
            response = req_json.get("response", "")
        except KeyError:
            resp.status_code = 400
            resp.media = {"error": "request json body should have 'text' key"}
            return

        # Generate text
        gen_text = self._generator.generate(context, response)

        # Set response
        resp.media = {"response": gen_text}


class ResponseGenerator:
    def __init__(self, model, tokenizer, max_len, top_p, top_k, num_cands):
        self._model = model
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._top_p = top_p
        self._top_k = top_k
        self._num_cands = num_cands

    def generate(self, context, response):
        input_builder = ChatLMModelInputBuilder(
            tokenizer=self._tokenizer,
            add_end_token=False,
        )
        # Prepare model input
        model_input = input_builder.build(
            context,
            response,
            batch_size=self._num_cands
        )

        # Prepare generator
        generator = TopPKGenerator(
            model=self._model,
            top_p=self._top_p,
            top_k=self._top_k,
            bad_ids=[0, 1],  # Ignore [PAD] and [UNK]
        )

        # Generate
        for _ in range(self._max_len):
            next_ids, _ = generator.step(
                input_ids=model_input["input_ids"],
                token_type_ids=model_input["token_type_ids"]
            )
            model_input = input_builder.update(model_input, next_ids)
            if input_builder.ended(model_input):
                break

        # Extract response
        res_ids = split_context_response_ids(
            model_input["input_ids"],
            self._tokenizer.sep_token_id,
            self._tokenizer.cls_token_id,
        )
        # Filter response
        res_ids = filter_out_by_coverage_ratio(res_ids, ratio=0.3)

        # Create response
        responses = [
            self._tokenizer.decode(r_ids).replace(" ", "")
            for _, r_ids in res_ids
        ]
        return responses[0]


def main(model_dir, address=None, port=None, max_len=100,
         top_p=0.95, top_k=50, num_cands=1):
    # Build API
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    gen = ResponseGenerator(model, tokenizer, max_len, top_p, top_k, num_cands)
    handler = APIHandler(generator=gen)
    api = responder.API()
    api.add_route("/chat", handler.generate)

    # Run API
    with torch.set_grad_enabled(False):
        api.run(address=address, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
