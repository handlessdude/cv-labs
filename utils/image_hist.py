from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from utils.fs import save_plot


@njit(cache=True)
def describe_channels(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    r_intensities, g_intensities, b_intensities = np.zeros(256), np.zeros(256), np.zeros(256)
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            ri, gi, bi = img[row][col]
            r_intensities[ri] += 1
            g_intensities[gi] += 1
            b_intensities[bi] += 1
    return r_intensities, g_intensities, b_intensities


@njit
def normalized_hists(img_in: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    r_hist, g_hist, b_hist = describe_channels(img_in)
    k = img_in.shape[0] * img_in.shape[1]  # for every hist: sum of bins = k
    return r_hist / k, g_hist / k, b_hist / k


def plot_channel_hists(img: np.ndarray, dir_out: str, filename_out: str):
    r_intensities, g_intensities, b_intensities = describe_channels(img)
    plt.plot(range(256), r_intensities, 'r')
    plt.plot(range(256), g_intensities, 'g')
    plt.plot(range(256), b_intensities, 'b')
    save_plot(dir_out, filename_out)

