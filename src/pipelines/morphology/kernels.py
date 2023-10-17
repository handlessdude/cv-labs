import numpy as np

BLACK, WHITE = 0, 255
EPS = 0.01


HOLE_RADIUS = 48
HOLE_DIAMETER = 2 * HOLE_RADIUS


def get_hole_mask():
    hole_mask = np.zeros((HOLE_DIAMETER, HOLE_DIAMETER), dtype=np.uint8)

    for row in range(HOLE_DIAMETER):
        for col in range(HOLE_DIAMETER):
            y = row - HOLE_RADIUS
            x = col - HOLE_RADIUS
            if y**2 + x**2 < HOLE_RADIUS**2 + EPS:
                hole_mask[row][col] = WHITE

    return hole_mask


def get_hole_ring():
    hole_ring = get_hole_mask()
    for row in range(HOLE_DIAMETER):
        for col in range(HOLE_DIAMETER):
            y = row - HOLE_RADIUS
            x = col - HOLE_RADIUS
            if (
                y**2 + x**2 < (HOLE_RADIUS - 1) ** 2 + EPS
                or y**2 + x**2 > HOLE_RADIUS**2 + EPS
            ):
                hole_ring[row][col] = BLACK
    return hole_ring


def get_disk(radius: int):
    diameter = radius * 2 + 1
    disk = np.zeros((diameter, diameter), dtype=np.uint8)
    for row in range(diameter):
        for col in range(diameter):
            y = row - radius
            x = col - radius
            if y**2 + x**2 < radius**2 + EPS:
                disk[row][col] = WHITE

    return disk


GEAR_BODY_RADIUS = 125
GEAR_BODY_DIAMETER = 2 * GEAR_BODY_RADIUS


def get_gear_body():
    gear_body = np.zeros((GEAR_BODY_DIAMETER, GEAR_BODY_DIAMETER), dtype=np.uint8)

    for row in range(GEAR_BODY_DIAMETER):
        for col in range(GEAR_BODY_DIAMETER):
            y = row - GEAR_BODY_RADIUS
            x = col - GEAR_BODY_RADIUS
            if y**2 + x**2 < GEAR_BODY_RADIUS**2 + EPS:
                gear_body[row][col] = WHITE

    return gear_body
