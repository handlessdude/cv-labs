import numpy as np
from numba import njit, prange
from typing import Callable

cfs_v1 = np.array([0.3, 0.59, 0.11], dtype=np.float32)


@njit(cache=True, inline='always')
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
    k = np.dot(cfs_v1.astype(np.float32), pixel.astype(np.float32))  # 0.3 * R + 0.59 * G + 0.11 * B
    return np.array([k, k, k]).astype(np.uint8)


@njit(parallel=True, cache=True)
def convert_to_grayscale(
  img_in: np.ndarray,
  img_out: np.ndarray,
  model: Callable[[np.ndarray], np.ndarray],
  use_bgr: bool = False
) -> np.ndarray:
    copied: np.ndarray = img_in[:, :, ::-1] if use_bgr else np.copy(img_in)
    for row in prange(0, copied.shape[0]):
        for col in prange(0, copied.shape[1]):
            img_out[row][col] = model(copied[row][col])
    return img_out
