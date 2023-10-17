import cv2
import numpy as np


def close(img: np.ndarray, kernel: np.ndarray):
    return cv2.erode(cv2.dilate(img, kernel), kernel)


def open(img: np.ndarray, kernel: np.ndarray):
    return cv2.dilate(cv2.erode(img, kernel), kernel)


def subtract(first: np.ndarray, second: np.ndarray):
    height, width, _ = first.shape
    img_out = np.zeros_like(first)
    for row in range(height):
        for col in range(width):
            for i in range(3):
                img_out[row][col][i] = max(first[row][col][i] - second[row][col][i], 0)

    return img_out


def OR(first: np.ndarray, second: np.ndarray):
    return np.maximum(first, second)


def AND(first: np.ndarray, second: np.ndarray):
    return np.minimum(first, second)
