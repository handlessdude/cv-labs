import numpy as np

from src.pipelines.frequency_filtering.kernels import get_ideal_filter
from src.pipelines.frequency_filtering.methods import (
    normalize_to_uint8,
)


def log_of_abs(arr_in: np.ndarray):
    return np.log(np.abs(arr_in))


def get_spectrum_from_img(img_in: np.ndarray):
    # code piece actually from https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    spectrum = log_of_abs(fshift)
    return f, fshift, spectrum


IDEAL_FILTER_R = 28


def fft_smoothen_sharpen(img_in: np.ndarray):
    f, fshift, spectrum = get_spectrum_from_img(img_in)

    ideal_filter = get_ideal_filter(*img_in.shape, IDEAL_FILTER_R)

    ideal_spectrum = fshift * ideal_filter
    img_temp = np.fft.ifftshift(ideal_spectrum)
    img_temp = np.fft.ifft2(img_temp)
    ideal_image = np.real(img_temp)

    ideal_filter_normalized = normalize_to_uint8(
        ideal_filter
    )  # have to do it bcuz we used np.float32 type

    ideal_image_normalized = normalize_to_uint8(ideal_image)
    _, _, ideal_image_spectrum = get_spectrum_from_img(ideal_image_normalized)
    ideal_image_spectrum = normalize_to_uint8(ideal_image_spectrum)

    inverse_ideal_filter = 1 - ideal_filter
    img_temp = fshift * inverse_ideal_filter
    img_temp = np.fft.ifftshift(img_temp)
    img_temp = np.fft.ifft2(img_temp)
    sharpened_ideal_image = np.real(img_temp)

    sharp_ideal_filter_normalized = normalize_to_uint8(
        inverse_ideal_filter
    )  # have to do it bcuz we used np.float32 type
    sharp_ideal_image_normalized = normalize_to_uint8(sharpened_ideal_image)
    _, _, sharp_ideal_image_spectrum = get_spectrum_from_img(
        sharp_ideal_image_normalized
    )
    sharp_ideal_image_spectrum = normalize_to_uint8(sharp_ideal_image_spectrum)

    return {
        "spectrum": normalize_to_uint8(spectrum),
        "smoothing": {
            "filter": ideal_filter_normalized,
            "spectrum": ideal_image_spectrum,
            "img_out": ideal_image_normalized,
        },
        "sharpening": {
            "filter": sharp_ideal_filter_normalized,
            "spectrum": sharp_ideal_image_spectrum,
            "img_out": sharp_ideal_image_normalized,
        },
    }
