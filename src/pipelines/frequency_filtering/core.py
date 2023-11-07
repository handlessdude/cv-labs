import numpy as np

from src.pipelines.frequency_filtering.kernels import get_ideal_filter
from src.pipelines.frequency_filtering.methods import (
    normalize_to_uint8,
)

EPS = 1e-10
IDEAL_FILTER_R = 30


def replace_zeros(img_in: np.ndarray):
    return np.where(img_in == 0, EPS, img_in)


def log_of_abs(arr_in: np.ndarray):
    return np.log(np.abs(arr_in))


def get_spectrum(img_in: np.ndarray):
    base_img = replace_zeros(img_in.astype(np.float32))
    dft = np.fft.fft2(base_img)
    dft_shift = np.fft.fftshift(dft)  # here applies filter
    spectrum = log_of_abs(dft_shift)
    return dft_shift, spectrum


def apply_filter(img_in: np.ndarray, filter: np.ndarray):
    dft_shift, img_in_spectrum = get_spectrum(img_in)

    filter_applied = dft_shift * filter

    dft_ishift = np.fft.ifftshift(filter_applied)
    idft = np.fft.ifft2(dft_ishift)
    img_out = np.abs(idft)
    _, img_out_spectrum = get_spectrum(img_out)

    return img_in_spectrum, img_out_spectrum, img_out


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
