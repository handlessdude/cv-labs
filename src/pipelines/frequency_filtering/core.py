import numpy as np


def get_magnitude_spectrum(img_in: np.ndarray):
    # code piece actually from https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    return magnitude_spectrum


def fft_smoothen_sharpen(img_in: np.ndarray):
    return {"spectrum": get_magnitude_spectrum(img_in)}
