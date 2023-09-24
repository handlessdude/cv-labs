import numpy as np
from PIL import Image
from utils.grayscale import convert_to_grayscale, to_gray_v1
import os


inputs_dir = "./inputs"
outputs_dir = "./outputs"


def make_path(filename: str, inputs_dir: str):
    return inputs_dir + '/' + filename


def save_img(img: np.ndarray, dir: str, filename: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    Image.fromarray(img.astype(np.uint8)).save(make_path(filename, dir))


filename_in = "red-hibiscus.jpg"
filename_out = "red-hibiscus-test.jpg"

img_in = np.asarray(Image.open(make_path(filename_in, inputs_dir)))
img_out = np.copy(img_in)
converted = convert_to_grayscale(img_in, img_out, to_gray_v1)
save_img(converted, outputs_dir, filename_out)

# first_img = convert_to_grayscale(filename, 'res1.png', to_gray_v1)
# second_img = convert_to_grayscale(filename, 'res2.png', to_gray_v2)
# pics_diff(first_img, second_img, "diff_res.png")
