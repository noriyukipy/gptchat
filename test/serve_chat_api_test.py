import pytest
import responder
from gptchat.cmd.serve_chat_api import ChatHandler


class MockPredictor:
    def predict(self, context):
        if context == "Hello":
            return "World"
        else:
            return "What?"


@pytest.fixture
def api():
    api = responder.API()
    api.add_route("/chat", ChatHandler(predictor=MockPredictor()).handle)
    print("Serve HTTP")
    yield api
    print("Close HTTP")


def test_handler_parse_param():
    handler = ChatHandler(predictor=MockPredictor())
    req_dict = {"context": "Hello"}
    res = handler.parse_param(req_dict=req_dict)
    assert res.context == "Hello", "want {}, got {}".format("Hello", res.context)


def test_handler_parse_param_error():
    handler = ChatHandler(predictor=MockPredictor())
    req_dict = {"text": "Hello"}
    with pytest.raises(KeyError):
        handler.parse_param(req_dict=req_dict)


def test_api(api):
    resp = api.requests.post("/chat", json={"context": "Hello"})
    assert resp.status_code == 200, f"unexpected status code, got {resp.status_code} want 200"
    assert resp.json() == {"context": "Hello", "response": "World"}


def test_response_with_invalid_json_request(api):
    resp = api.requests.post("/chat", json={"text": "Hello"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "request json body should have 'context' key"