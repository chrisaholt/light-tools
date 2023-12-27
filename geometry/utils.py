import numpy as np

def expand_shape(array: np.array):
    """Expand array's shape to be one more dimension."""
    return array.reshape((1,) + array.shape)
