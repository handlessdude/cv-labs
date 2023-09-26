# 1. Цветовая коррекция изображений s. 18
#   1.1 Коррекция с опорным цветом [x]
#   1.2 Серый мир [x]
#   1.3 По виду функции преобразования [x]
# 2. Яркостная коррекция в интерактивном режиме по виду функции преобразования (необязательное дополнительное задание)
#   2.1 График функции кусочно линейный
#   2.2 График функции интерполируется сплайном
# 3. Коррекция на основе гистограммы
#   3.1 Нормализация гистограммы
#   3.2 Эквализация гистограммы


from utils.fs import open_img, save_img, make_path
from resources import files_for_color_correction as data
from utils.image_hist import plot_channel_hists
import numpy as np
from numba import njit, prange
from typing import Callable


dir_in = "../inputs/02"
dir_out = "../outputs/02"


@njit(parallel=True, cache=True)
def gray_world_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    means = np.zeros(3)
    for row in prange(0, img_in.shape[0]):
        for col in prange(0, img_in.shape[1]):
            for i in range(3):
                means[i] += img_in[row][col][i]
    means *= 1 / (img_in.shape[0] * img_in.shape[1])
    avg = means.sum() * (1 / 3)
    k = np.array([avg / ch_mean for ch_mean in means]).astype(np.float32)
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = img_in[row][col] * k

    return img_out


@njit(parallel=True, cache=True)
def reference_color_correction(
    img_in: np.ndarray,
    img_out: np.ndarray,
    dst: np.ndarray,
    src: np.ndarray,
) -> np.ndarray:
    k = dst / src
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = img_in[row][col] * k
    return img_out.astype(np.uint8)


@njit(parallel=True, cache=True)
def linear_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    ymin = np.array([np.inf, np.inf, np.inf])
    ymax = np.array([-np.inf, -np.inf, -np.inf])
    for row in prange(0, img_in.shape[0]):
        for col in prange(0, img_in.shape[1]):
            for i in range(3):
                ymin[i] = min(ymin[i], img_in[row][col][i])
                ymax[i] = max(ymax[i], img_in[row][col][i])

    cfs = np.array([255, 255, 255]) / (ymax - ymin)
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = (img_in[row][col] - ymin) * cfs

    return img_out.astype(np.uint8)


@njit(parallel=True, cache=True)
def logarithmic_correction(img_in: np.ndarray, img_out: np.ndarray, k: np.float32) -> np.ndarray:
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            img_out[row][col] = k * np.log(img_in[row][col] + 1)

    return img_out.astype(np.uint8)


from utils.image_hist import describe_channels


@njit
def normalized_hists(img_in: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    r_hist, g_hist, b_hist = describe_channels(img_in)
    k = img_in.shape[0] * img_in.shape[1]  # for every hist: sum of bins = k
    return r_hist / k, g_hist / k, b_hist / k


# TODO redo
@njit
def make_lut(src_hist: np.ndarray, dst_hist: np.ndarray):
    ymin = src_hist.min()
    ymax = src_hist.max()
    return ((dst_hist - ymin) * 255 / (ymax - ymin)).astype(np.uint8)

# TODO redo
@njit(parallel=True, cache=True)
def normalization_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    r_hist, g_hist, b_hist = describe_channels(img_in)
    r_norm, g_norm, b_norm = normalized_hists(img_in)

    r_lut = make_lut(r_hist, r_norm)
    g_lut = make_lut(g_hist, g_norm)
    b_lut = make_lut(b_hist, b_norm)

    lut = [r_lut, g_lut, b_lut]
    for row in prange(0, img_out.shape[0]):
        for col in prange(0, img_out.shape[1]):
            for i in range(3):
                img_out[row][col][i] = lut[i][img_in[row][col][i]]

    return img_out.astype(np.uint8)


# TODO redo
def equalize_channel(channel: np.ndarray):
    # Calculate the histogram of the channel
    hist, bins = np.histogram(channel, bins=256, range=(0, 256))
    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalize the CDF to have values between 0 and 255
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    # Interpolate the channel values based on the normalized CDF
    equalized_channel = np.interp(channel, bins[:-1], cdf_normalized)
    return equalized_channel.astype(np.uint8)


# TODO redo
def equalization_correction(img_in: np.ndarray, img_out: np.ndarray) -> np.ndarray:
    # Split the image into its Red, Green, and Blue channels
    r, g, b = img_in[:, :, 0], img_in[:, :, 1], img_in[:, :, 2]
    # Apply histogram equalization to each channel
    r_eq = equalize_channel(r)
    g_eq = equalize_channel(g)
    b_eq = equalize_channel(b)
    # Stack the equalized channels to form the corrected image
    img_out = np.dstack((r_eq, g_eq, b_eq))
    return img_out


def make_correction(
    img_in: np.ndarray,
    model: Callable[[np.ndarray, ...], np.ndarray],
    dir_out: str,
    filename_out: str,
    *args
):
    img_out = np.copy(img_in)
    model(img_in, img_out, *args)
    save_img(img_out, dir_out, filename_out)
    return img_out


def main():
    print('Process start...')
    fns_to_use = set([equalization_correction])
    for entry, model in zip(data, [
        gray_world_correction,
        reference_color_correction,
        linear_correction,
        logarithmic_correction,
        normalization_correction,
        equalization_correction
    ]):
        if not model in fns_to_use:
            print('skipping model')
            continue
        img_in = open_img(dir_in, entry['in'])
        if img_in.shape[2] == 4:
            img_in = img_in[:, :, 0:3]
        subfolder = make_path(dir_out, entry['subfolder'])
        img_out = make_correction(
            img_in,
            model,
            subfolder,
            entry['out'],
          *entry['additional_args']
        )
        plot_channel_hists(img_in, subfolder, entry['in_hist'])
        plot_channel_hists(img_out, subfolder, entry['out_hist'])
    print('Done!')


if __name__ == '__main__':
    main()
