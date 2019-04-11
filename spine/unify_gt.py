import numpy as np
import os
from os import listdir
import nibabel as nib
import argparse
import sys
from skimage import io
from skimage import color
from natsort import natsorted
import matplotlib
from scipy import ndimage
import scipy

'''
script to unify spine and dendrite ground truths

inputs:
 - spines ground truth nii
 - dendrite ground truth tiff

outputs:
 - unified ground truth

execution example:
 - python3 unify_gt.py --path_spine "../../data/raw/spine_seg_tiff" --path_dendrite "../../data/raw/dendrite_seg_tiff" --path_out "../../data/raw/aa"

'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_spine', help='path to the spine gt folder.')
    parser.add_argument('--path_dendrite', help='path to the dendrite gt folder.')
    parser.add_argument('--path_out', help='path to the output folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_spine = parsed_args.path_spine  # get spine folder
    path_dendrite = parsed_args.path_dendrite  # get dendrite folder
    path_out = parsed_args.path_out  # get output folder
    os.mkdir(path_out)

    dir_dendrite = listdir(path_dendrite)
    dir_dendrite = natsorted(dir_dendrite)

    for case_folder in dir_dendrite:

        dir_dendrite_case = listdir(os.path.join(path_dendrite, case_folder))
        dir_dendrite_case = natsorted(dir_dendrite_case)

        renamed_case_folder, name2 = case_folder.split("_")  # get name

        os.mkdir(os.path.join(path_out, renamed_case_folder))

        dir_spines_case = listdir(os.path.join(path_spine, renamed_case_folder + "_spGT"))
        dir_spines_case = natsorted(dir_spines_case)

        for slice in dir_dendrite_case:

            indx_slice = dir_dendrite_case.index(slice)
            data_spine_slice = io.imread(os.path.join(path_spine, renamed_case_folder + "_spGT", dir_spines_case[indx_slice]))

            data_spine_slice = color.rgb2gray(data_spine_slice)

            pos_spine = np.where(data_spine_slice != [0])

            for pos in range(len(pos_spine[0])):
                data_spine_slice[pos_spine[0][pos], pos_spine[1][pos]] = 255

            data_dendritic_slice = io.imread(os.path.join(path_dendrite, case_folder, slice))
            pos_dendrite = np.where(data_dendritic_slice != [0])

            for pos in range(len(pos_dendrite[0])):
                data_spine_slice[pos_dendrite[0][pos], pos_dendrite[1][pos]] = 150

            scipy.misc.imsave(path_out + "/" + renamed_case_folder + "/" + str(dir_spines_case[indx_slice]), data_spine_slice)


if __name__ == "__main__":
    main()
