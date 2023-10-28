import numpy as np
import cv2

from src.pipelines.color_correction.methods import gamma_correction

laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


def enhance_skeletons(img_in: np.ndarray):
    laplacian = cv2.filter2D(src=img_in, ddepth=-1, kernel=laplacian_kernel)
    sharpened_1 = cv2.add(img_in, laplacian)
    sobel = cv2.convertScaleAbs(cv2.Sobel(sharpened_1, cv2.CV_64F, 1, 0))
    smoothed = cv2.blur(sobel, ksize=(3, 3))
    multiplied = cv2.multiply(sharpened_1, smoothed, scale=1 / 255)
    sharpened_2 = cv2.add(img_in, multiplied)
    clarified = gamma_correction(sharpened_2, 0.5)
    return [
        laplacian,
        sharpened_1,
        sobel,
        smoothed,
        multiplied,
        sharpened_2,
        clarified,
    ]
