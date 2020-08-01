import os
import numpy as np
import random
import tensorflow as tf
from envyaml import EnvYAML


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
