# 1. Цветовая коррекция изображений s. 18
#   1.1 Коррекция с опорным цветом
#   1.2 Серый мир
#   1.3 По виду функции преобразования
# 2. Яркостная коррекция в интерактивном режиме по виду функции преобразования (необязательное дополнительное задание)
#   2.1 График функции кусочно линейный
#   2.2 График функции интерполируется сплайном
# 3. Коррекция на основе гистограммы
#   3.1 Нормализация гистограммы
#   3.2 Эквализация гистограммы


from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from utils.fs import make_path, open_img, save_plot
from resources import filename_data


dir_in = "../inputs"
dir_out = "../outputs-02"


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


def main():
    for data in filename_data:
        img = open_img(dir_in, data['in'])
        plot_channel_hists(img, dir_out,  data['out'])


if __name__ == '__main__':
    main()
