import numpy as np

from src.pipelines.frequency_filtering.kernels import (
    get_ideal_filter,
    get_butterworth_filter,
    get_gaussian_filter,
)
from src.pipelines.frequency_filtering.methods import (
    normalize_to_uint8,
    apply_filter,
)


IDEAL_FILTER_R = 30


def conduct_filtering_pipeline(img_in: np.ndarray, filter: np.ndarray):
    img_in_spec, img_out_spec, img_out = apply_filter(img_in, filter)

    inverse_filter = 1 - filter
    _, iimg_out_spec, iimg_out = apply_filter(img_in, inverse_filter)

    return {
        "spectrum": normalize_to_uint8(img_in_spec),
        "smoothing": {
            "filter": normalize_to_uint8(filter),
            "spectrum": normalize_to_uint8(img_out_spec),
            "img_out": normalize_to_uint8(img_out),
        },
        "sharpening": {
            "filter": normalize_to_uint8(inverse_filter),
            "spectrum": normalize_to_uint8(iimg_out_spec),
            "img_out": normalize_to_uint8(iimg_out),
        },
    }


def apply_ideal_filter(img_in: np.ndarray):
    filter = get_ideal_filter(*img_in.shape, IDEAL_FILTER_R)
    return conduct_filtering_pipeline(img_in, filter)


def apply_butterworth_filter(img_in: np.ndarray):
    filter = get_butterworth_filter(*img_in.shape, IDEAL_FILTER_R, 2)
    return conduct_filtering_pipeline(img_in, filter)


def apply_gaussian_filter(img_in: np.ndarray):
    filter = get_gaussian_filter(*img_in.shape, IDEAL_FILTER_R)
    return conduct_filtering_pipeline(img_in, filter)
