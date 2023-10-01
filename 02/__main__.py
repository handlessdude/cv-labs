from utils.io.fs import open_img, save_img, make_path
from resources import files_for_color_correction as data
from utils.image_hist import plot_channel_hists
from utils.color_correction import *

dir_in = "../inputs/02"
dir_out = "../outputs/02"


def main():
    print('Process start...')
    fns_to_use = set([
        gray_world_correction,
        reference_color_correction,
        linear_correction,
        logarithmic_correction,
        normalization_correction,
        equalization_correction
    ])
    for entry, model in zip(data, [
        gray_world_correction,
        reference_color_correction,
        linear_correction,
        logarithmic_correction,
        normalization_correction,
        equalization_correction
    ]):
        if not model in fns_to_use:
            print('skipping model')
            continue
        img_in = open_img(dir_in, entry['in'])
        if img_in.shape[2] == 4:
            img_in = img_in[:, :, 0:3]
        subfolder = make_path(dir_out, entry['subfolder'])
        img_out = make_correction(
            img_in,
            model,
            subfolder,
            entry['out'],
          *entry['additional_args']
        )
        plot_channel_hists(img_in, subfolder, entry['in_hist'])
        plot_channel_hists(img_out, subfolder, entry['out_hist'])
    print('Done!')


if __name__ == '__main__':
    main()
