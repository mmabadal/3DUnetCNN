import argparse
import sys
from skimage import io
import numpy as np
import os
from os import listdir
import nibabel as nib
import matplotlib

'''
script to add layers to the nii until reaching an specified number

inputs:
 - grey scale and ground truth nii
 - number of target layers
 
outputs:
 - grey scale and ground truth nii mith layers added

execution example:
 - python3 nii_extend.py --path_data "../../data/dendrite_spine_nii" --path_out "../../data/dendrite_spine_nii_extended" --z_stack 112
'''

# python3 nii_extend.py --path_data "/disk/3d_unet/data/new_v2/spine_cils_nii" --path_out "/disk/3d_unet/data/new_v2/spine_cils_nii_extended" --z_stack 112


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', help='path to the input case folders.')
    parser.add_argument('--path_out', help='path to export the output case folders.')
    parser.add_argument('--z_stack', default=112, type=int, help='number of target slices')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_data = parsed_args.path_data  # get path in
    path_out = parsed_args.path_out   # get path out
    z_stack = parsed_args.z_stack  # get target slices

    os.mkdir(path_out)

    dir = listdir(path_data)

    black_slice = np.zeros((1024, 1024), dtype=np.uint8)  # aux black slice

    for case_folder in dir:  #for each case

        os.mkdir(os.path.join(path_out, case_folder))

        # load data files
        data_file = os.path.join(path_data, case_folder, "spine.nii.gz")
        data_image = nib.load(data_file)
        data = data_image.get_data()

        # stack black slice 'indx' times
        indx = data.shape[2]

        print("case folder: " + case_folder)
        print("idx data:" + str(indx))
        while indx != z_stack:
            data = np.dstack((data, black_slice))
            indx += 1

        # Save extended data
        data_file_out = os.path.join(path_out, case_folder, "spine.nii.gz")
        data = nib.Nifti1Image(data, affine=np.eye(4, 4))
        nib.save(data, data_file_out)

        # load GT files
        truth_file = os.path.join(path_data, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        # stack black slice 'indx' times
        indx = truth.shape[2]
        print("idx gt:" + str(indx))
        while indx != z_stack:
            truth = np.dstack((truth, black_slice))
            indx += 1

        # Save extended GT
        truth_file_out = os.path.join(path_out, case_folder, "truth.nii.gz")
        truth = nib.Nifti1Image(truth, affine=np.eye(4, 4))
        nib.save(truth, truth_file_out)


if __name__ == "__main__":
    main()
