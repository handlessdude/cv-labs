import numpy as np
from numba import njit, prange

from src.utils.arrays import cartesian_product


FOREGROUND_IDX = 2
palette = np.array(
    [
        np.array([0, 0, 0]),  # 0 no-class area
        np.array([0, 0, 255]),  # 1 background
        np.array([255, 0, 0]),  # 2 foreground
    ]
)


def get_markings_mask(markings: np.ndarray) -> np.ndarray:
    mask_out = np.zeros((markings.shape[0], markings.shape[1]))
    for row in prange(mask_out.shape[0]):
        for col in prange(mask_out.shape[1]):
            for idx, color in list(enumerate(palette)):
                if np.array_equal(markings[row][col], color):
                    mask_out[row][col] = idx
    return mask_out.astype(np.uint8)


def apply_mask(img_in: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_out = img_in.copy()
    for row in range(img_out.shape[0]):
        for col in range(img_out.shape[1]):
            if not np.array_equal(mask[row][col], FOREGROUND_IDX):
                img_out[row][col] = np.array([0, 0, 0])
    return img_out


GROWCUT_ITERATIONS = 50
theta = np.array([0, 0.5, 0.5])
eps = 1e-5
indices = np.array(range(-2, 3))
deltas = cartesian_product(indices, indices)


@njit(parallel=True)
def get_growcut_mask(img_in: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width, _ = img_in.shape
    for _ in range(GROWCUT_ITERATIONS):
        for row in prange(2, height - 2):
            for col in prange(2, width - 2):
                winner = mask[row, col]
                a_max = 0
                for drow, dcol in deltas:
                    if not mask[row, col] and not mask[row + drow, col + dcol]:
                        continue
                    norm = np.linalg.norm(
                        (
                            img_in[row, col]
                            - img_in[
                                row + drow,
                                col + dcol,
                            ]
                        ).astype(np.float32)
                    )
                    a = 1 / (norm + eps)
                    if a + eps > a_max:
                        winner = mask[row + drow][col + dcol]
                        a_max = a
                if winner and theta[winner] * a_max > theta[mask[row, col]]:
                    mask[row, col] = winner
    return mask


def growcut(img_in: np.ndarray, markings: np.ndarray):
    markings_mask = get_markings_mask(markings)
    growcut_mask = get_growcut_mask(img_in, markings_mask)
    return apply_mask(img_in, growcut_mask)
