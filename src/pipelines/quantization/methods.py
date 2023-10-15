import numpy as np
from numba import njit, prange
import cv2

from src.pipelines.grayscale.methods import convert_to_grayscale_v1
from src.pipelines.image_description.methods import describe_channels, describe_channel
from src.utils.arrays import get_first_nonzero, get_last_nonzero

halftone_cfs = np.array([0.0721, 0.7154, 0.2125], dtype=np.float32)


@njit(parallel=True, cache=True)
def convert_to_halftone(img_in: np.ndarray):
    img_out = np.zeros(img_in.shape, dtype=np.uint8)
    print(img_in.shape)
    for j in prange(0, img_in.shape[0]):
        for i in prange(0, img_in.shape[1]):
            img_out[j][i] = np.dot(
                halftone_cfs.astype(np.float32), img_in[j][i].astype(np.float32)
            )
    return img_out.astype(np.uint8)


def convert_to_quantitized(img_in: np.ndarray, levels: np.ndarray):
    grayscale_img = convert_to_grayscale_v1(img_in)
    levels = np.sort(levels)

    lut = np.concatenate(
        tuple(
            [
                np.repeat(level, level if idx == 0 else level - levels[idx - 1])
                for idx, level in enumerate(levels)
            ]
        )
    )
    if len(lut) < 256:
        lut = np.concatenate((lut, np.repeat(255, 256 - len(lut))))

    quantitized_img = cv2.LUT(grayscale_img, lut)
    return quantitized_img.astype(np.uint8)


def get_otsu_threshold(
    grayscale_channel: np.ndarray, lb: np.uint8 = 1, ub: np.uint8 = 255
):
    hist = describe_channel(grayscale_channel)
    prob = hist / hist.sum()

    lower_i = np.max([lb, get_first_nonzero(hist)])
    upper_i = np.min([ub, get_last_nonzero(hist)])

    expectation = 0
    for i in range(lower_i + 1, upper_i + 1):
        expectation += i * prob[i]

    best_threshold = lower_i

    w0, w1 = prob[lower_i], 1 - prob[lower_i]
    mean0, mean1 = lower_i, expectation / w1

    max_variance = w0 * w1 * (mean0 - mean1) ** 2

    for i in range(lower_i + 1, upper_i):
        mean0 = mean0 * w0 + i * prob[i]
        mean1 = mean1 * w1 - i * prob[i]

        w0 += prob[i]
        w1 = 1 - w0

        mean0 = mean0 / w0
        mean1 = mean1 / w1

        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2

        if between_class_variance > max_variance:
            best_threshold = i
            max_variance = between_class_variance

    return best_threshold


def binarize(img_in: np.ndarray, threshold: np.uint8):
    below_threshold = np.repeat(0, threshold)
    above_threshold = np.repeat(255, 256 - threshold)
    lut = np.concatenate((below_threshold, above_threshold), axis=0)
    binarized_img = cv2.LUT(img_in, lut)
    return binarized_img.astype(np.uint8)


def otsu_global_binarization(img_in: np.ndarray):
    grayscale_img = convert_to_grayscale_v1(img_in)
    grayscale_channel = grayscale_img[:, :, 0]
    return binarize(grayscale_img, get_otsu_threshold(grayscale_channel))


def otsu_hierarchical_binarization(img_in: np.ndarray):
    return otsu_global_binarization(img_in)
