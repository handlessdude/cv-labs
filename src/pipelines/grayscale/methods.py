import numpy as np
from numba import njit


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


def convert_to_grayscale(img_as_array: np.ndarray, model, use_bgr=False):
    converted_img = np.array(
        [
            np.array([model(pixel[::-1] if use_bgr else pixel) for pixel in line])
            for line in img_as_array
        ]
    )
    return converted_img


def pics_diff(first_img_in: np.ndarray, second_img_in: np.ndarray):
    x = first_img_in.astype(np.float32) - second_img_in.astype(np.float32)
    converted_img = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)
    return converted_img
