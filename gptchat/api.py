from fastapi import FastAPI
from pydantic import BaseModel
import typing


class Request(BaseModel):
    context: str
    response: str = None


class ModelInfo(BaseModel):
    output: typing.Any


class Response(BaseModel):
    response: str
    request: Request
    model_info: ModelInfo


def build_api(handler):
    app = FastAPI(
        title="GPTChat",
        description="",
        version="0.0.0",

    )
    app.add_api_route("/generate", handler.generate, methods=["POST"])
    return app
