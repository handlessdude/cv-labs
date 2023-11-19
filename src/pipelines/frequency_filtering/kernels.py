import math

import numpy as np

from src.utils.arrays import vec_len


def get_ideal_filter(shape0: int, shape1: int, r: int):
    kernel = np.zeros((shape0, shape1), dtype=np.float32)
    center = np.array([shape0 * 0.5, shape1 * 0.5])
    for row in range(shape0):
        for col in range(shape1):
            if vec_len(center, np.array([row, col])) <= r:
                kernel[row][col] = 1
    return kernel


def get_butterworth_filter(shape0: int, shape1: int, r: int, n: int):
    kernel = np.zeros((shape0, shape1), dtype=np.float32)
    center = np.array([shape0 * 0.5, shape1 * 0.5])
    for row in range(shape0):
        for col in range(shape1):
            kernel[row][col] = 1 / (1 + vec_len(center, np.array([row, col])) / r) ** (
                2 * n
            )
    return kernel


def get_gaussian_filter(shape0: int, shape1: int, r: int):
    res = np.zeros((shape0, shape1), dtype=np.float32)
    center = np.array([shape0 * 0.5, shape1 * 0.5])
    for row in range(shape0):
        for col in range(shape1):
            res[row][col] = math.exp(
                (-1 * vec_len(center, np.array([row, col])) ** 2) / (2 * r**2)
            )
    return res
