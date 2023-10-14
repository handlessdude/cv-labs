import numpy as np
from numba import njit, prange
from typing import Callable


cfs_v1 = np.array([0.3, 0.59, 0.11], dtype=np.float32)


@njit(cache=True, inline="always")
def to_gray_v1(pixel: np.ndarray) -> np.ndarray:
    """
    converts given pixel of [r, g, b] form to grayscale
    Parameters
    ----------
    pixel : np.ndarray
        The first number.
    Returns
    -------
    np.ndarray
        pixel converted to grayscale
    """
    k = np.dot(
        cfs_v1.astype(np.float32), pixel.astype(np.float32)
    )  # 0.3 * R + 0.59 * G + 0.11 * B
    return np.array([k, k, k]).astype(np.uint8)


def map_pixels(
    img_in: np.ndarray, model: Callable[[np.ndarray], np.ndarray], use_bgr=False
):
    converted_img = np.array(
        [
            np.array([model(pixel[::-1] if use_bgr else pixel) for pixel in line])
            for line in img_in
        ]
    )
    return converted_img


@njit(parallel=True, cache=True)
def convert_to_grayscale_v1(img_in: np.ndarray):
    height, width, _ = img_in.shape
    channel = np.zeros((height, width), dtype=np.uint8)
    for row in prange(0, height):
        for col in prange(0, width):
            channel[row][col] = to_gray_v1(img_in[row][col])[0]
    return np.dstack((channel, channel, channel))


def pics_diff(first_img_in: np.ndarray, second_img_in: np.ndarray):
    x = first_img_in.astype(np.float32) - second_img_in.astype(np.float32)
    converted_img = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)
    return converted_img
