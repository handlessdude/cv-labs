import cv2
from src.utils.fs_io import make_path
from src.pipelines.grayscale.methods import (
    convert_to_grayscale_v1,
    to_gray_v1,
    pics_diff,
)


def run_motion_detector(inputs_dir: str, video_in: str, outputs_dir: str):
    video_path = make_path(video_in, inputs_dir)

    video_capture = cv2.VideoCapture(video_path)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    frame_rate = int(video_capture.get(5))

    output_video = cv2.VideoWriter(
        make_path(outputs_dir, "motion_sensor.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (frame_width, frame_height),
    )

    prev_frame = None

    while True:
        ret, curr_frame = video_capture.read()

        if not ret:
            break

        curr_frame = convert_to_grayscale_v1(curr_frame, to_gray_v1, True)

        if prev_frame is None:
            prev_frame = curr_frame
            continue

        frame_diff = pics_diff(prev_frame, curr_frame)

        output_video.write(frame_diff)

        prev_frame = curr_frame

    video_capture.release()
    output_video.release()

    cv2.destroyAllWindows()
