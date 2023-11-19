from typing import List

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2


def make_path(dir_in: str, filename_in: str):
    return os.path.join(dir_in, filename_in)


def save_img(img: np.ndarray, dir_out: str, filename_out: str):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    Image.fromarray(img.astype(np.uint8)).save(make_path(dir_out, filename_out))


def save_plot(dir_out: str, filename_out: str):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    plt.savefig(make_path(dir_out, filename_out))
    plt.cla()
    plt.clf()


def open_img(dir_in: str, filename_in: str):
    img_as_array = np.asarray(Image.open(make_path(dir_in, filename_in)))
    if len(img_as_array.shape) <= 2:
        return channel_to_img(img_as_array)
    elif img_as_array.shape[2] == 4:
        img_as_array = img_as_array[:, :, 0:3]
    return img_as_array


def img_to_base64(img_in: np.ndarray, format: str = "png"):
    buffer_in = BytesIO()
    Image.fromarray(img_in).save(buffer_in, format=format.upper())
    return "data:image/png;base64," + base64.b64encode(buffer_in.getvalue()).decode(
        "utf-8"
    )


def base64_to_ndarray(base64_str: str) -> np.ndarray:
    # Remove the header (e.g., "data:image/png;base64,") and decode
    encoded_data = base64_str.split(',')[1]
    decoded_bytes = base64.b64decode(encoded_data)

    # Convert to a NumPy array
    img_array = np.frombuffer(decoded_bytes, dtype=np.uint8)

    # Decode the image using OpenCV
    img = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def channels_to_img(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    return np.dstack((r, g, b))


def channel_to_img(gs_channel: np.ndarray):
    return channels_to_img(gs_channel, gs_channel, gs_channel)


def frames_to_channel(frames: List[np.ndarray]):
    return np.vstack(tuple(frames))
