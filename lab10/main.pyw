import cv2
import numpy as np


# mask smoothing kernel
def get_kernel(kernel_r: int = 20):
    kernel_d = kernel_r * 2
    kernel_shape = (kernel_d, kernel_d)
    kernel_center = (kernel_r, kernel_r)
    thickness = -1  # we do now want ellipses
    color = 1
    kernel = cv2.circle(
        np.zeros(kernel_shape, dtype=np.uint8),
        kernel_center,
        kernel_r,
        color,
        thickness,
    )
    return kernel


video_in_path = "./breakdance_in.mp4"
video_out_path = "./breakdance_out.mp4"


def main():
    video_capture = cv2.VideoCapture(video_in_path)

    if not video_capture.isOpened():
        print("Error opening video stream or file")

    frame_shape = (int(video_capture.get(4)), int(video_capture.get(3)))
    frame_rate = int(video_capture.get(5))

    out = cv2.VideoWriter(
        video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_shape, True
    )

    first_gs = np.zeros(frame_shape, dtype=np.uint8)
    second_gs = np.zeros(frame_shape, dtype=np.uint8)
    kernel = get_kernel()
    first_frame = True

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, second_gs)

        if first_frame:
            first_frame = False
            np.copyto(first_gs, second_gs)
            continue

        mask = (second_gs.astype(np.float32) - first_gs.astype(np.float32)).astype(
            np.uint8
        )

        cv2.equalizeHist(mask, mask)

        mean = mask.mean()
        mask[mask < mean] = 0
        mask[mask >= mean] = 1

        # mask smoothing
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
        res = frame * mask[:, :, np.newaxis]

        cv2.imshow("Object detection", res)
        out.write(res)

        np.copyto(first_gs, second_gs)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# https://www.youtube.com/watch?v=W1ValKNYw_U&ab_channel=V%CE%9EG%CE%9B
