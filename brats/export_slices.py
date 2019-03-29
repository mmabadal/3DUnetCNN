import numpy as np
import nibabel as nib
import os
import glob
import argparse
import sys
from os import listdir
from scipy import ndimage
import scipy


'''
script to export an image of each slice of nii file

inputs: 
 - nii stack
outputs:
 - images of each slice

execution example:
 - python3 export_slices.py --path_data "../../data/spines_nii" --path_out "../../data/raw/original_grey_tiff_sliced" --name "spine"
 
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', help='path to the input case folders .')
    parser.add_argument('--path_out', help='path to export the images.')
    parser.add_argument('--name', help='working files name.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_data = parsed_args.path_data  # get data path
    path_out = parsed_args.path_out  # get output path
    name = parsed_args.name  # get output path
    os.mkdir(path_out)

    file = name + ".nii.gz"

    dir = listdir(path_data)


    for case_folder in dir:  # for each case

        os.mkdir(os.path.join(path_out, case_folder))

        # load nii
        spine_file = os.path.join(path_data, case_folder, file)
        spine_image = nib.load(spine_file)
        spine = spine_image.get_data()

        # extract and save slices
        for slice in range(spine.shape[2]):
            spine_slice = spine[..., slice]
            scipy.misc.imsave(path_out + "/" + case_folder + "/" + str(slice) + "_spine.tiff", spine_slice)


if __name__ == "__main__":
    main()
