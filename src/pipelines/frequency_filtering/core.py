import numpy as np

from src.pipelines.frequency_filtering.kernels import get_ideal_filter
from src.pipelines.frequency_filtering.methods import (
    normalize_to_uint8,
    apply_filter,
)


IDEAL_FILTER_R = 30


def apply_ideal_filter(img_in: np.ndarray):
    ideal_filter = get_ideal_filter(*img_in.shape, IDEAL_FILTER_R)
    img_in_spec, img_out_spec, img_out = apply_filter(img_in, ideal_filter)

    inverse_ideal_filter = 1 - ideal_filter
    _, iimg_out_spec, iimg_out = apply_filter(img_in, inverse_ideal_filter)

    return {
        "spectrum": normalize_to_uint8(img_in_spec),
        "smoothing": {
            "filter": normalize_to_uint8(ideal_filter),
            "spectrum": normalize_to_uint8(img_out_spec),
            "img_out": normalize_to_uint8(img_out),
        },
        "sharpening": {
            "filter": normalize_to_uint8(inverse_ideal_filter),
            "spectrum": normalize_to_uint8(iimg_out_spec),
            "img_out": normalize_to_uint8(iimg_out),
        },
    }
