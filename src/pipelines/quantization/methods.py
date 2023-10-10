import numpy as np
from numba import njit, prange
import cv2


halftone_cfs = np.array([0.0721, 0.7154, 0.2125], dtype=np.float32)


@njit(parallel=True, cache=True)
def convert_to_halftone(img_in: np.ndarray):
    img_out = np.zeros(img_in.shape, dtype=np.uint8)
    for j in prange(0, img_in.shape[0]):
        for i in prange(0, img_in.shape[1]):
            img_out[j][i] = np.dot(
                halftone_cfs.astype(np.float32), img_in[j][i].astype(np.float32)
            )

    # fig, ax = plt.subplots()
    # ax.imshow(img_out, cmap="gray")
    # fig.canvas.draw()
    # pixel_data = np.array(fig.canvas.renderer._renderer)
    # if pixel_data.shape[2] == 4:
    #     pixel_data = pixel_data[:, :, 0:3]
    return img_out.astype(np.uint8)


def convert_to_quantitized(img_in: np.ndarray, levels: np.ndarray):
    levels = np.sort(levels)
    lut = np.zeros(0, dtype=np.uint8)
    for j in range(len(levels)):
        lut = np.append(
            lut, [np.repeat(levels[j], levels[j] - (0 if j == 0 else levels[j - 1]))]
        )
    if len(lut) < 256:
        lut = np.append(lut, [np.repeat(255, 256 - len(lut))])
    qF = cv2.LUT(img_in, lut)
    return qF.astype(np.uint8)
