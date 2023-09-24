# 1. Загрузка изображения
# 2. Преобразование к полутоновому изображению двумя способами:
# 1) Gray=0.3*R+0.59*G+0.11*B;
# 2) Gray=(R + G + B)/3,
# 3. Получить разность этих полутоновых изображений
# 4. Взять фрагмент видео и покадрово обрабатывать:
#   1) преобразовывать к полутоновому
#   2) получать разность между соседними кадрами,. Типа примитивнейший датчик движения))

import os
import sys
import numpy as np
from PIL import Image
inputs_dir = "./inputs"
outputs_dir = "./outputs"


def make_path(filename, inputs_dir):
    return inputs_dir + '/' + filename


def save_img(img, filename):
    temp = Image.fromarray(img.astype(np.uint8))
    temp.save(filename)


def to_gray_v1(pixel):
    R, G, B = pixel[0], pixel[1], pixel[2],
    k = 0.3 * R + 0.59 * G + 0.11 * B
    return [k for _ in range(3)]


def to_gray_v2(pixel):
    R, G, B = pixel[0], pixel[1], pixel[2],
    gamma = 1 / 3
    k = gamma * R + gamma * G + gamma * B
    return np.array([k for _ in range(3)]).astype(np.uint8)


def convert_to_grayscale(filename_in, filename_out, model):
    img = Image.open(make_path(filename_in, inputs_dir))
    img_as_array = np.asarray(img)
    converted_img = np.array([np.array([model(pixel) for pixel in line]) for line in img_as_array])
    save_img(converted_img, make_path(filename_out, outputs_dir))
    return converted_img


def pics_diff(first_img_in, second_img_in, filename_out):
    x = first_img_in.astype(np.float32) - second_img_in.astype(np.float32)
    converted_img = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)
    save_img(converted_img, make_path(filename_out, outputs_dir))
    return converted_img


filename = "red-hibiscus.jpg"
first_img = convert_to_grayscale(filename, 'res1.png', to_gray_v1)
second_img = convert_to_grayscale(filename, 'res2.png', to_gray_v2)
pics_diff(first_img, second_img, "diff_res.png")
