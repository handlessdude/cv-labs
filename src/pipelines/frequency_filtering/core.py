import numpy as np


def get_magnitude_spectrum(img_in: np.ndarray):
    # code piece actually from https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    print(magnitude_spectrum)
    return magnitude_spectrum


def shit_to_ok(img_in: np.ndarray):
    img_in_normalized = (img_in - np.min(img_in)) / (np.max(img_in) - np.min(img_in))
    return (img_in_normalized * 255).astype(np.uint8)


def fft_smoothen_sharpen(img_in: np.ndarray):
    spectrum = get_magnitude_spectrum(img_in)
    return {"spectrum": shit_to_ok(spectrum)}
