from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from utils.fs import save_plot


@njit(parallel=True, cache=True)
def describe_channels(img: np.ndarray):
    r_intensities, g_intensities, b_intensities = np.zeros(256), np.zeros(256), np.zeros(256)
    for row in prange(0, img.shape[0]):
        for col in prange(0, img.shape[1]):
            ri, gi, bi = img[row][col][0], img[row][col][1], img[row][col][2]
            r_intensities[ri] += 1
            g_intensities[gi] += 1
            b_intensities[bi] += 1
    return r_intensities, g_intensities, b_intensities


def plot_channel_hists(img: np.ndarray, dir_out: str, filename_out: str):
    r_intensities, g_intensities, b_intensities = describe_channels(img)
    plt.plot(range(256), r_intensities, 'r')
    plt.plot(range(256), g_intensities, 'g')
    plt.plot(range(256), b_intensities, 'b')
    save_plot(dir_out, filename_out)
