import numpy as np
from numba import njit, prange
import cv2

from src.pipelines.grayscale.methods import convert_to_grayscale_v1
from src.pipelines.image_description.methods import describe_channel
from src.utils.arrays import get_first_nonzero, get_last_nonzero
from src.utils.fs_io import frames_to_channel, channel_to_img

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
    levels = np.sort(np.unique(levels))
    print("levels", levels)
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

    grayscale_img = convert_to_grayscale_v1(img_in)
    quantitized_img = cv2.LUT(grayscale_img, lut)
    return quantitized_img.astype(np.uint8)


def get_otsu_threshold_sosi(
    grayscale_channel: np.ndarray,
    hist_lb: np.uint8 = 0,
    hist_ub: np.uint8 = 255,
) -> np.uint8:
    hist = describe_channel(grayscale_channel)
    left_edge = max(hist_lb, get_first_nonzero(hist))
    right_edge = min(hist_ub, get_last_nonzero(hist))

    prob = hist / hist[left_edge : right_edge + 1].sum()

    best_threshold, max_variance = 0, 0

    for threshold in range(left_edge, right_edge + 1):
        prob_below = prob[left_edge:threshold]
        prob_above = prob[threshold : right_edge + 1]

        w0 = prob_below.sum()
        w1 = prob_above.sum()

        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.dot(np.arange(left_edge, threshold), prob_below) / w0
        mean1 = np.dot(np.arange(threshold, right_edge + 1), prob_above) / w1

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
    return binarize(grayscale_img, get_otsu_threshold_sosi(grayscale_channel))


def otsu_local_binarization(img_in: np.ndarray, y_delims: np.ndarray) -> np.ndarray:
    """
    :param img_in: single channel of some image, e.g. channel's shape is (h, w)
    :param frames: array of rects, e.g. ((y,x), (y,x)), where first element is top left corner
    :return: channel's intensity histogram
    """
    grayscale_channel = convert_to_grayscale_v1(img_in)[:, :, 0]

    height, width = grayscale_channel.shape

    clip = np.vectorize(
        lambda x, lower_bound, upper_bound: max(lower_bound, min(upper_bound, x))
    )

    y_delims = np.unique(
        np.concatenate(
            (np.array([0]), np.sort(clip(y_delims, 0, height)), np.array([height]))
        )
    )
    y_intervals = list(zip(y_delims[:-1], y_delims[1:]))
    frames = [grayscale_channel[y0:y1, :] for y0, y1 in y_intervals]
    frames_count = len(frames)
    frames_out = [None for _ in range(frames_count)]

    for idx in prange(frames_count):
        frames_out[idx] = binarize(frames[idx], get_otsu_threshold_sosi(frames[idx]))

    channel_out = frames_to_channel(frames_out)
    img_out = channel_to_img(channel_out)
    return img_out


def otsu_hierarchical_step(
    channel: np.ndarray,
    depth: int = 3,
    hist_lb: np.uint8 = 0,
    hist_ub: np.uint8 = 255,
    min_interval: int = 20,
):
    if depth == 0 or (hist_ub - hist_lb) <= min_interval:
        return []
    t = get_otsu_threshold_sosi(channel, hist_lb, hist_ub)
    thresholds_below = otsu_hierarchical_step(
        channel, depth - 1, hist_lb, t - 1, min_interval
    )
    thresholds_above = otsu_hierarchical_step(
        channel, depth - 1, t + 1, hist_ub, min_interval
    )

    levels = np.sort(
        np.unique(np.concatenate((thresholds_below, np.array([t]), thresholds_above)))
    )
    return levels


def otsu_hierarchical_binarization(img_in: np.ndarray):
    grayscale_channel = convert_to_grayscale_v1(img_in)[:, :, 0]
    levels = otsu_hierarchical_step(grayscale_channel)
    return convert_to_quantitized(img_in, levels)
