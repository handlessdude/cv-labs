# 1. Загрузить изображение
# 2. Построить гистограмму по каждому из каналов (RGB)

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

inputs_dir = "../assets"

def main():
    if len(sys.argv) < 2:
        return

    filename = inputs_dir + "/" + sys.argv[1]
    if not os.path.isfile(filename):
        print('No file {} found in current directory.'.format(filename))
        return

    img = Image.open(filename)
    converted_img = np.asarray(img)
    r_intensities = [0 for _ in range(256)]
    g_intensities = [0 for _ in range(256)]
    b_intensities = [0 for _ in range(256)]
    for line in converted_img:
        for pixel in line:
            r_intensities[pixel[0]] += 1
            g_intensities[pixel[1]] += 1
            b_intensities[pixel[2]] += 1
    plt.plot(range(256), r_intensities, 'r')
    plt.plot(range(256), g_intensities, 'g')
    plt.plot(range(256), b_intensities, 'b')
    plt.savefig(inputs_dir + '/result.png')
