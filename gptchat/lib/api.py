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
