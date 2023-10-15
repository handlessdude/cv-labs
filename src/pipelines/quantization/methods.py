import numpy as np
from numba import njit, prange
import cv2

from src.pipelines.grayscale.methods import convert_to_grayscale_v1
from src.pipelines.image_description.methods import describe_channel

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


def get_otsu_threshold(grayscale_channel: np.ndarray) -> int:
    prob = describe_channel(grayscale_channel)
    prob /= prob.sum()

    best_threshold, max_variance = 0, 0

    for threshold in range(256):
        w0 = prob[:threshold].sum()
        w1 = prob[threshold:].sum()

        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.dot(np.arange(threshold), prob[:threshold]) / w0
        mean1 = np.dot(np.arange(threshold, 256), prob[threshold:]) / w1

        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            best_threshold = threshold

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


# @njit(parallel=True, cache=True)
def otsu_local_binarization(img_in: np.ndarray, window_size: int = 5) -> np.ndarray:
    grayscaled = convert_to_grayscale_v1(img_in)[:, :, 0]

    height, width = grayscaled.shape
    output = np.zeros_like(grayscaled)

    for y in prange(height):
        for x in prange(width):
            # Calculate the local region for the current pixel
            x1, x2, y1, y2 = (
                x - window_size,
                x + window_size + 1,
                y - window_size,
                y + window_size + 1,
            )

            # Ensure the region is within the image boundaries
            x1, x2, y1, y2 = max(0, x1), min(width, x2), max(0, y1), min(height, y2)

            # Extract the local region
            local_region = grayscaled[y1:y2, x1:x2]

            # Calculate the local Otsu threshold
            local_threshold = get_otsu_threshold(local_region)

            # Apply the threshold to the current pixel
            if grayscaled[y, x] > local_threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0

    return np.dstack((output, output, output))


def otsu_hierarchical_binarization(img_in: np.ndarray):
    return otsu_global_binarization(img_in)
