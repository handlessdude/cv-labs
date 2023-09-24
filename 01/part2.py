import numpy as np
from PIL import Image
import cv2

inputs_dir = "./inputs"
outputs_dir = "./outputs"


def make_path(filename, dir):
    return dir + '/' + filename


def save_img(img, filename):
    temp = Image.fromarray(img.astype(np.uint8))
    temp.save(filename)


coeffs_v1 = np.array([0.3, 0.59, 0.11], dtype=np.float32)


def to_gray_v1(pixel: np.ndarray):  # [r g b]
    k = np.dot(coeffs_v1, pixel)  # 0.3 * R + 0.59 * G + 0.11 * B
    return np.array([k, k, k]).astype(np.uint8)


gamma = 1 / 3

coeffs_v2 = np.array([gamma, gamma, gamma], dtype=np.float32)


def to_gray_v2(pixel: np.ndarray):
    # R, G, B = pixel[0], pixel[1], pixel[2]
    # k = gamma * R + gamma * G + gamma * B
    k = np.dot(coeffs_v2, pixel)
    return np.array([k, k, k]).astype(np.uint8)


def convert_to_grayscale(img_as_array: np.ndarray, model, use_bgr=False):
    converted_img = np.array([
        np.array(
            [model(pixel[::-1] if use_bgr else pixel) for pixel in line]
        ) for line in img_as_array
    ])
    return converted_img


def pics_diff(first_img_in: np.ndarray, second_img_in: np.ndarray):
    x = first_img_in.astype(np.float32) - second_img_in.astype(np.float32)
    converted_img = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)
    return converted_img


video_path = make_path('cat_breakdance.mp4', inputs_dir)


video_capture = cv2.VideoCapture(video_path)


frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
frame_rate = int(video_capture.get(5))


output_video = cv2.VideoWriter(make_path('motion_sensor.mp4', outputs_dir),
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               frame_rate,
                               (frame_width, frame_height))


prev_frame = None


while True:
    ret, curr_frame = video_capture.read()

    if not ret:
        break

    curr_frame = convert_to_grayscale(curr_frame, to_gray_v1, True)

    if prev_frame is None:
        prev_frame = curr_frame
        continue

    frame_diff = pics_diff(prev_frame, curr_frame)

    output_video.write(frame_diff)

    prev_frame = curr_frame

video_capture.release()
output_video.release()

cv2.destroyAllWindows()

# https://www.youtube.com/watch?v=W1ValKNYw_U&ab_channel=V%CE%9EG%CE%9B
