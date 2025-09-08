import numpy as np


def scale(array: np.ndarray):
    """
    Scales a NumPy array to the [0, 1] range.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array - min_val) / (max_val - min_val)
    return scaled, min_val, max_val


def descale(scaled_array: np.ndarray, min_val: float, max_val: float):
    """
    Reverts a scaled array back to its original range.
    """
    return scaled_array * (max_val - min_val) + min_val