from __future__ import print_function
import cv2
import numpy as np


WINNAME = 'HELLO ANYONE HERE'
FILE_OUT = 'sosi.mp4'


def make_tracker(frame, bounding_box):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bounding_box)
    return tracker

def clip_bounds(lower: int, upper: int, lower_bound: int, upper_bound: int):
    return np.clip(np.array([lower, upper]), lower_bound, upper_bound)


def grabcut(frame, face):
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    (x, y, w, h) = face
    (dside, dup, ddown) = (int(w * 0.3), int(h * 0.3), int(h))

    top_left = (max(0, x - dside), max(y - dup, 0))
    bottom_right = (min(frame.shape[1], x + w + dside), min(frame.shape[0], y + h + ddown))
    (x_c, y_c) = (x + w // 2 - top_left[0], y + h // 2 - top_left[1])

    mask = np.zeros((bottom_right[1] - top_left[1], bottom_right[0] - top_left[0]), np.uint8)
    for dy in range(-h // 2, h // 2):
        for dx in range(-w // 2, w // 2):
            dy_normalized = dy / h
            dx_normalized = dx / w
            if pow(dy_normalized, 2) + pow(dx_normalized, 2) < 1:
                y0 = y_c + dy
                x0 = x_c + dx
                if (0 <= y0 < mask.shape[0] and 0 <= x0 < mask.shape[1]):
                    mask[y0, x0] = 255

    frame_cropped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.grabCut(frame_cropped, mask, (x - top_left[0], y - top_left[1], w, h), background_model, foreground_model, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    tl_x, tl_y = top_left
    br_x, br_y = bottom_right
    frame[tl_y:br_y, tl_x:br_x] = frame_cropped * mask2[:, :, np.newaxis]
    frame[br_y:] = 0
    frame[:tl_y] = 0
    frame[:, :tl_x] = 0
    frame[:, br_x:] = 0
    return frame

# 1. Выделить лицо с помощью каскадов Хаара, пример использования из документации OpenCV
def detect_and_display(frame, tracker, out):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # 2. Вместо вызова детектора лица для каждого кадра,
    # можно добавить отслеживание позиции найденного лица,
    # и запускать детекцию только в случае, когда трекер лица сбился.
    if tracker:
        succ, face = tracker.update(frame_gray)
        if succ:
            # FACE === x,y,w,h
            frame = grabcut(frame, face)
        else:
            return detect_and_display(frame, None)
    else:
        faces = face_cascade.detectMultiScale(frame_gray)
        if len(faces) > 0:
            face = faces[0]
            tracker = make_tracker(frame_gray, face)
            frame = grabcut(frame, face)
        else:
            frame *= 0

    cv2.imshow(WINNAME, frame)
    out.write(frame)
    return tracker, frame


face_cascade_name = 'haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'haarcascade_eye_tree_eyeglasses.xml'
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()


#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)


#-- 2. Read the video stream
camera_device = 0  # webcam.
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_shape = (frame_height, frame_width)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(FILE_OUT,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      frame_rate,
                      frame_shape
                      )

WAIT_KEY_DELAY = 10
ESC_CODE = 27

tracker = None
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    tracker, new_frame = detect_and_display(frame, tracker, out)

    if cv2.waitKey(WAIT_KEY_DELAY) == ESC_CODE:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
