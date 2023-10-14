import numpy as np
from numba import njit, prange
import cv2

from src.pipelines.grayscale.methods import convert_to_grayscale_v1
from src.pipelines.image_description.methods import describe_channels

halftone_cfs = np.array([0.0721, 0.7154, 0.2125], dtype=np.float32)


@njit(parallel=True, cache=True)
def convert_to_halftone(img_in: np.ndarray):
    img_out = np.zeros(img_in.shape, dtype=np.uint8)
    for j in prange(0, img_in.shape[0]):
        for i in prange(0, img_in.shape[1]):
            img_out[j][i] = np.dot(
                halftone_cfs.astype(np.float32), img_in[j][i].astype(np.float32)
            )
    return img_out.astype(np.uint8)


def convert_to_quantitized(img_in: np.ndarray, levels: np.ndarray):
    levels = np.sort(levels)
    lut = np.zeros(0, dtype=np.uint8)
    for j in range(len(levels)):
        lut = np.append(
            lut, [np.repeat(levels[j], levels[j] - (0 if j == 0 else levels[j - 1]))]
        )
    if len(lut) < 256:
        lut = np.append(lut, [np.repeat(255, 256 - len(lut))])
    qF = cv2.LUT(img_in, lut)
    return qF.astype(np.uint8)


def first_of(np_array: np.ndarray, cond):
    for i in range(np_array.shape[0]):
        if cond(np_array[i]):
            return i
    return None


def last_of(np_array, cond):
    for i in range(np_array.shape[0])[::-1]:
        if cond(np_array[i]):
            return i
    return None


def get_otsu_threshold(img_in, __min=1, __max=255):
    temp_hist, _, _ = describe_channels(img_in)

    _min = max(__min, first_of(temp_hist, lambda x: x > 0))
    _max = min(__max, last_of(temp_hist, lambda x: x > 0))

    hist = temp_hist / sum(temp_hist[_min : (_max + 1)])

    m = 0
    for t in range(_min + 1, _max + 1):
        m += t * hist[t]

    threshold = _min

    q1 = hist[_min]
    q2 = 1 - q1
    mu1 = _min  # * hist[_min] / q1
    mu2 = m / q2

    maxSigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)

    for t in range(_min + 1, _max):
        mu1 = mu1 * q1 + t * hist[t]
        mu2 = mu2 * q2 - t * hist[t]

        q1 += hist[t]
        q2 = 1 - q1

        mu1 = mu1 / q1
        mu2 = mu2 / q2

        newSigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if maxSigma < newSigma:
            threshold = t
            maxSigma = newSigma
    return threshold


def binarize(img_in: np.ndarray, threshold: np.uint8):
    below_threshold = np.repeat(0, threshold)
    above_threshold = np.repeat(255, 256 - threshold)
    lut = np.concatenate((below_threshold, above_threshold), axis=0)
    binarized_img = cv2.LUT(img_in, lut)
    return binarized_img.astype(np.uint8)


def otsu_global_binarization(img_in: np.ndarray):
    grayscale_img = convert_to_grayscale_v1(img_in)
    return binarize(grayscale_img, get_otsu_threshold(grayscale_img))


def otsu_hierarchical_binarization(img_in: np.ndarray):
    return otsu_global_binarization(img_in)
