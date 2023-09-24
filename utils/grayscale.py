import numpy as np
from numba import njit


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
