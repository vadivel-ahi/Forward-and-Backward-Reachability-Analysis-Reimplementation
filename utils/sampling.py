# utils/sampling.py
import numpy as np
from fbra.boxes import Box

def sample_box(box: Box, n_samples):
    return np.random.uniform(box.low, box.up, size=(n_samples, len(box.low)))
