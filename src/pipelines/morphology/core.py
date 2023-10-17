import numpy as np
import cv2

from src.pipelines.morphology.kernels import (
    get_gear_body,
    get_hole_ring,
    get_hole_mask,
    get_disk,
)

from src.pipelines.morphology.methods import OR, open, AND, close, subtract


def inspect_gears(img_in: np.ndarray):
    hole_ring = get_hole_ring()
    hole_mask = get_hole_mask()
    gear_body = get_gear_body()
    sampling_ring_spacer = get_disk(4)
    sampling_ring_width = get_disk(8)
    tip_spacing = get_disk(9)
    defect_cue = get_disk(25)

    b = cv2.erode(img_in, hole_ring)
    c = cv2.dilate(b, hole_mask)
    d = OR(img_in, c)

    ## BEGIN SAMPLING_RING
    d_no_teeth = open(d, gear_body)
    d_brought_to_teeth_base = cv2.dilate(d_no_teeth, sampling_ring_spacer)
    d_brought_to_teeth_tip = cv2.dilate(d_brought_to_teeth_base, sampling_ring_width)
    e = subtract(d_brought_to_teeth_tip, d_brought_to_teeth_base)
    ## END

    f = AND(e, img_in)
    g = cv2.dilate(f, tip_spacing)
    # sr_sub_g = subtract(e, g)
    sr_sub_g = cv2.subtract(e, g)  # dont know if its allowed by lab conditions
    h = OR(cv2.dilate(sr_sub_g, defect_cue), g)

    return [b, c, d, d_no_teeth, d_brought_to_teeth_base, e, f, g, sr_sub_g, h]
