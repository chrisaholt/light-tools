import numpy as np

def indicator_func(
        t: np.array,
        thresh: np.array):
    """
    Returns nan if t < thresh and 1 if t >= thresh.
    """
    indicator_array = np.ones(t.shape )
    indicator_array[t < thresh] = np.nan
    return indicator_array

def scale_func(
    optical_distance_to_point: np.array,
):
    return 1 / (1 + optical_distance_to_point)
