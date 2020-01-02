from transformers import GPT2Tokenizer
from transformers import BertJapaneseTokenizer
import random
import numpy as np
import torch
import os


def set_seed(seed, num_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu):
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def to_device(device, tensors):
    return [tensor.to(device) for tensor in tensors]


def save(config, model, tokenizer, save_dir):
    """Save config, model and tokenizer to save_dir directory
    If save_dir does not exist, this functions automatically creates
    it.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def build_tokenizer(model):
    """
    Args:
        model (str): model name or directory path to model files
    """
    tokenizer_classes = [BertJapaneseTokenizer, GPT2Tokenizer]

    tokenizer = None
    for cls in tokenizer_classes:
        try:
            tokenizer = cls.from_pretrained(model)
            # [TODO] Replace logging with test
            import logging
            logging.info(f"Loaded {tokenizer.__class__.__name__} with {model}")
        except (ValueError, UnboundLocalError, TypeError):
            pass
    if not tokenizer:
        raise Exception("Tokenizer build error")

    return tokenizer
