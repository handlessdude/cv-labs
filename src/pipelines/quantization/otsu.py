import numpy as np
from numba import jit, prange
from src.pipelines.image_description.methods import describe_channel
from src.pipelines.grayscale.methods import convert_to_grayscale_v1


@jit(cache=True)
def otsu_thresholding(image: np.ndarray) -> int:
    # Calculate a local histogram
    histogram = describe_channel(image)

    # Variables to store best threshold and best between-class variance
    best_threshold = 0
    best_variance = 0

    for threshold in range(256):
        # Class probabilities
        w0 = histogram[:threshold].sum()
        w1 = histogram[threshold:].sum()

        if w0 == 0 or w1 == 0:
            continue

        # Class means (weighted)
        mean0 = np.dot(np.arange(threshold), histogram[:threshold]) / w0
        mean1 = np.dot(np.arange(threshold, 256), histogram[threshold:]) / w1

        # Between-class variance
        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2

        # Check if the variance is greater than the best found so far
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = threshold

    return best_threshold


@jit(parallel=True, cache=True)
def otsu_local_binarization(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    grayscaled = convert_to_grayscale_v1(image)[:, :, 0]
    height, width = grayscaled.shape
    output = np.zeros_like(grayscaled)

    for y in prange(height):
        for x in prange(width):
            # Calculate the local region for the current pixel
            x1, x2, y1, y2 = (
                x - window_size,
                x + window_size + 1,
                y - window_size,
                y + window_size + 1,
            )

            # Ensure the region is within the image boundaries
            x1, x2, y1, y2 = max(0, x1), min(width, x2), max(0, y1), min(height, y2)

            # Extract the local region
            local_region = grayscaled[y1:y2, x1:x2]

            # Calculate the local Otsu threshold
            local_threshold = otsu_thresholding(local_region)

            # Apply the threshold to the current pixel
            if grayscaled[y, x] > local_threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0

    return np.dstack((output, output, output))
