from src.models.schemas.fs_io import IOEntry
import numpy as np

files_for_color_correction = [
    IOEntry(
        subfolder="gray_world",
        filename_in="gray_world.png",
        filename_out="gray_world.png",
        hist_in="gray_world_in_hist.png",
        hist_out="gray_world_out_hist.png",
        additional_args=[],
    ),
    IOEntry(
        subfolder="reference_color",
        filename_in="reference_color.png",
        filename_out="reference_color.png",
        hist_in="reference_color_in_hist.png",
        hist_out="reference_color_out_hist.png",
        additional_args=[
            np.array([1, 1, 255]),  # dst
            np.array([255, 1, 1]),  # src
        ],
    ),
    IOEntry(
        subfolder="linear",
        filename_in="linear.png",
        filename_out="linear.png",
        hist_in="linear_in_hist.png",
        hist_out="linear_out_hist.png",
        additional_args=[],
    ),
    IOEntry(
        subfolder="logarithm",
        filename_in="logarithm.png",
        filename_out="logarithm.png",
        hist_in="logarithm_in_hist.png",
        hist_out="logarithm_out_hist.png",
        additional_args=[35],
    ),
    IOEntry(
        subfolder="normalization",
        filename_in="normalization.png",
        filename_out="normalization.png",
        hist_in="normalization_in_hist.png",
        hist_out="normalization_out_hist.png",
        additional_args=[],
    ),
    IOEntry(
        subfolder="equalization",
        filename_in="normalization.png",
        filename_out="equalization.png",
        hist_in="equalization_in_hist.png",
        hist_out="equalization_out_hist.png",
        additional_args=[],
    ),
]
