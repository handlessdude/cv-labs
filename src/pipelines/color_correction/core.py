from src.utils.fs_io import open_img, make_path
from src.utils.image_hist import plot_channel_hists
from src.pipelines.color_correction.methods import *
from src.models.schemas.fs_io import IOEntry


def make_correction(
    img_in: np.ndarray, model: Callable[[np.ndarray, ...], np.ndarray], *args, **kwargs
):
    img_out = np.copy(img_in)
    model(img_in, img_out, *args)
    dir_out = kwargs.get("dir_out", None)
    filename_out = kwargs.get("filename_out", None)
    if dir_out and filename_out:
        save_img(img_out, dir_out, filename_out)
    return img_out


def run_color_corrections(dir_in: str, dir_out: str, data: list[IOEntry]):
    print("Process start...")
    fns_to_use = set(
        [
            gray_world_correction,
            reference_color_correction,
            linear_correction,
            logarithmic_correction,
            normalization_correction,
            equalization_correction,
        ]
    )
    for entry, model in zip(
        data,
        [
            gray_world_correction,
            reference_color_correction,
            linear_correction,
            logarithmic_correction,
            normalization_correction,
            equalization_correction,
        ],
    ):
        if not model in fns_to_use:
            print("skipping model")
            continue
        img_in = open_img(dir_in, entry["in"])
        if img_in.shape[2] == 4:
            img_in = img_in[:, :, 0:3]
        subfolder = make_path(dir_out, entry["subfolder"])
        img_out = make_correction(
            img_in,
            model,
            *entry["additional_args"],
            dir_out=subfolder,
            filename_out=entry["out"],
        )
        plot_channel_hists(img_in, subfolder, entry["in_hist"])
        plot_channel_hists(img_out, subfolder, entry["out_hist"])
    print("Done!")
