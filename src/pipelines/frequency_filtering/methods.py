import numpy as np


def normalize_to_uint8(img_in: np.ndarray):
    img_in_normalized = (img_in - np.min(img_in)) / (np.max(img_in) - np.min(img_in))
    return (img_in_normalized * 255).astype(np.uint8)


def vec_len(first: np.ndarray, second: np.ndarray):
    return np.linalg.norm(first - second)


EPS = 1e-10


def replace_zeros(img_in: np.ndarray):
    return np.where(img_in == 0, EPS, img_in)


def log_of_abs(arr_in: np.ndarray):
    return np.log(np.abs(arr_in))


def get_raw_spectrum(img_in: np.ndarray):
    base_img = replace_zeros(img_in.astype(np.float32))
    dft = np.fft.fft2(base_img)
    dft_shift = np.fft.fftshift(dft)  # here applies filter
    spectrum = log_of_abs(dft_shift)
    return dft_shift, spectrum


def get_spectrum(img_in: np.ndarray):
    _, img_in_spectrum = get_raw_spectrum(img_in)
    return normalize_to_uint8(img_in_spectrum)


def apply_filter(img_in: np.ndarray, filter: np.ndarray):
    dft_shift, img_in_spectrum = get_raw_spectrum(img_in)

    filter_applied = dft_shift * filter

    dft_ishift = np.fft.ifftshift(filter_applied)
    idft = np.fft.ifft2(dft_ishift)
    img_out = np.abs(idft)
    _, img_out_spectrum = get_raw_spectrum(img_out)

    return img_in_spectrum, img_out_spectrum, img_out
