import os
import numpy as np
import random
import tensorflow as tf
from envyaml import EnvYAML
from fastapi import FastAPI
from pydantic import BaseModel
import typing


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def load_yaml(path):
    """
    Args:
        path (str): File path of yaml configuration file
    Returns:
        Dict[str, Any]:
    """
    return EnvYAML(path, include_environment=False).export()


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

