from utils.io.fs import save_img
import numpy as np
from numba import njit, prange
from typing import Callable
from utils.image_hist import normalized_hists


@njit(parallel=True, cache=True)
def gray_world_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    means = np.zeros(3)
    for row in prange(0, img_in.shape[0]):
        for col in prange(0, img_in.shape[1]):
            for i in range(3):
                means[i] += img_in[row][col][i]
    means *= 1 / (img_in.shape[0] * img_in.shape[1])
    avg = means.sum() * (1 / 3)
    k = np.array([avg / ch_mean for ch_mean in means]).astype(np.float32)
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = img_in[row][col] * k

    return img_out


@njit(parallel=True, cache=True)
def reference_color_correction(
    img_in: np.ndarray,
    img_out: np.ndarray,
    dst: np.ndarray,
    src: np.ndarray,
) -> np.ndarray:
    k = dst / src
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = img_in[row][col] * k
    return img_out.astype(np.uint8)


@njit(parallel=True, cache=True)
def linear_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    ymin = np.array([np.inf, np.inf, np.inf])
    ymax = np.array([-np.inf, -np.inf, -np.inf])
    for row in range(0, img_in.shape[0]):
        for col in range(0, img_in.shape[1]):
            for i in range(3):
                ymin[i] = min(ymin[i], img_in[row][col][i])
                ymax[i] = max(ymax[i], img_in[row][col][i])

    cfs = np.array([255, 255, 255]) / (ymax - ymin)
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = (img_in[row][col] - ymin) * cfs

    return img_out.astype(np.uint8)


@njit(parallel=True, cache=True)
def logarithmic_correction(img_in: np.ndarray, img_out: np.ndarray, k: np.float32) -> np.ndarray:
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = k * np.log(img_in[row][col] + 1)

    return img_out.astype(np.uint8)



# todo check
@njit
def make_lut(xp: np.ndarray, fp: np.ndarray):
    return np.interp([x for x in range(256)], xp, fp)


@njit(parallel=True, cache=True)
def normalization_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    imax, imin = -np.inf, np.inf
    for x in range(img_in.shape[0]):
        for y in range(img_in.shape[1]):
            for i in range(3):
                imax = max(imax, img_in[x][y][i])
                imin = min(imin, img_in[x][y][i])

    cfs = 255 / (imax - imin)

    for row in prange(0, img_in.shape[0]):
        for col in prange(0, img_in.shape[1]):
            img_out[row][col] = (img_in[row][col] - imin) * cfs

    return img_out


@njit(parallel=True, cache=True)
def equalization_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    r_norm, g_norm, b_norm = normalized_hists(img_in)
    r_cm, g_cm, b_cm = r_norm.cumsum(), g_norm.cumsum(), b_norm.cumsum()

    for row in prange(0, img_in.shape[0]):
        for col in prange(0, img_in.shape[1]):
            ri, gi, bi = img_in[row][col]
            img_out[row][col] = 255 * np.array([r_cm[ri], g_cm[gi], b_cm[bi]])

    return img_out


def make_correction(
  img_in: np.ndarray,
  model: Callable[[np.ndarray, ...], np.ndarray],
  dir_out: str,
  filename_out: str,
  *args
):
    img_out = np.copy(img_in)
    model(img_in, img_out, *args)
    save_img(img_out, dir_out, filename_out)
    return img_out
