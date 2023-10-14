from numba import njit, prange
from typing import Callable, List
import numpy as np


@njit(parallel=True, cache=True)
def map_arr(
  _in: np.ndarray,
  _out: np.ndarray,
  callback: Callable[[np.ndarray], np.ndarray],
  use_bgr: bool = False
) -> np.ndarray:
    copied: np.ndarray = _in[:, :, ::-1] if use_bgr else np.copy(_in)
    for row in prange(0, copied.shape[0]):
        for col in prange(0, copied.shape[1]):
            _out[row][col] = callback(copied[row][col])
    return _out


def get_first_nonzero(arr_in: np.ndarray):
    indices = np.flatnonzero(arr_in)
    return indices[0] if len(indices) else None


def get_last_nonzero(arr_in: np.ndarray):
    return 255 - get_first_nonzero(np.flip(arr_in))
