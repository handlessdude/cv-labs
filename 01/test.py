import numpy as np
from PIL import Image
import os
from functools import partial
from utils.grayscale import to_gray_v1
from utils.array_processing import map_arr


inputs_dir = "../inputs"
outputs_dir = "../outputs"


def make_path(filename: str, inputs_dir: str):
    return inputs_dir + '/' + filename


def save_img(img: np.ndarray, dir: str, filename: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    Image.fromarray(img.astype(np.uint8)).save(make_path(filename, dir))


filename_in = "red-hibiscus.jpg"
filename_out = "red-hibiscus-test.jpg"


convert_to_grayscale = partial(map_arr, callback=to_gray_v1)


img_in = np.asarray(Image.open(make_path(filename_in, inputs_dir)))
img_out = np.copy(img_in)
converted = convert_to_grayscale(img_in, img_out)
save_img(converted, outputs_dir, filename_out)
