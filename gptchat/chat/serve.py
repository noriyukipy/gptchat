import responder
from gptchat.lib import chat_utils
from gptchat.lib import train_utils
from transformers import GPT2DoubleHeadsModel


class ResponderModel:
    def __init__(self, model, tokenizer, max_len, top_p):
        self._model = model
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._top_p = top_p

    async def generate(self, req, resp):
        req_json = await req.media()
        text = req_json["text"]
        gen = chat_utils.generate(
            tokenizer=self._tokenizer,
            model=self._model,
            text=text,
            max_len=self._max_len,
            top_p=self._top_p
        )
        res_data = {
            "text": text,
            "model_output": gen,
            "reply": chat_utils.extract_reply(gen, self._tokenizer.unk_token)
        }
        resp.media = res_data


def main(address, port, model, max_len=30, top_p=0.9):
    # Initialze objects shared by all threads
    api = responder.API()
    tokenizer = train_utils.build_tokenizer(model)
    model = GPT2DoubleHeadsModel.from_pretrained(model)

    # Routing
    resp_model = ResponderModel(model, tokenizer, max_len, top_p)
    api.add_route("/generate", resp_model.generate)

    api.run(address=address, port=port)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
