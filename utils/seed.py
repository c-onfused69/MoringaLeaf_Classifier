import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """
    Set deterministic seeds for Python, NumPy, and TensorFlow to ensure reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Optional: configure TensorFlow to use deterministic operations if required
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"Set global random seed to {seed}")
