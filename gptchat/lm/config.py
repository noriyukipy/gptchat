from pydantic import BaseModel
from typing import List, Union


class ConfigInput(BaseModel):
    train_file: str
    valid_file: str
    tokenizer_file: str
    pretrained_model_dir: Union[None, str]


class ConfigOutput(BaseModel):
    model_dir: str
    tokenizer_file: str
    tensorboard_dir: str
    checkpoint_path: str


class ConfigTrain(BaseModel):
    block_size: int
    seed: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    warmup_rate: float
    patience: float


class ConfigPred(BaseModel):
    do_sample: bool
    seed: int
    max_length: int
    top_k: int
    top_p: float
    bad_words: List[str]


class ConfigModelParams(BaseModel):
    n_embd: int
    n_layer: int
    n_head: int
    n_ctx: int


class Config(BaseModel):
    input: ConfigInput
    output: ConfigOutput
    model_params: ConfigModelParams
    train: ConfigTrain
    pred: ConfigPred
