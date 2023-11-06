import numpy as np

from src.pipelines.frequency_filtering.methods import vec_len


def get_ideal_filter(shape0: int, shape1: int, r: int):
    kernel = np.zeros((shape0, shape1), dtype=np.float32)
    center = np.array([shape0 * 0.5, shape1 * 0.5])
    for row in range(shape0):
        for col in range(shape1):
            if vec_len(center, np.array([row, col])) <= r:
                kernel[row][col] = 1
    return kernel
