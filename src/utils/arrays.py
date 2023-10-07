from numba import njit, prange
from typing import Callable
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
