import numpy as np


def normalize_to_uint8(img_in: np.ndarray):
    img_in_normalized = (img_in - np.min(img_in)) / (np.max(img_in) - np.min(img_in))
    return (img_in_normalized * 255).astype(np.uint8)


def vec_len(first: np.ndarray, second: np.ndarray):
    return np.linalg.norm(first - second)
