import os
import glob
import numpy as np
import argparse
import nibabel as nib
from os import listdir
import sys
import nibabel as nib
from plyfile import PlyData, PlyElement

'''
script to convert a ply file into nii

inputs:
 - ply file

outputs:
 - nii file

execution example:
 - python3 load_ply.py --path_run "results/spines_dendrite/128x128x32_wloss_300ep" --name "dendrite_threshold.ply"
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--name', help='working file name')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get prediction folder
    name = parsed_args.name  # get file name

    file = name + ".ply"
    path_pred = os.path.join(path_run, "prediction")
    dir = listdir(path_pred)

    for case_folder in dir:  # for each case

        data_nii = np.zeros((1024, 1024, 112), dtype=np.uint8)  # nii data init

        # load file
        file_path = os.path.join(path_pred, case_folder, file)
        ply = PlyData.read(file_path)
        data_ply = ply.elements[0].data  # get data

        # pass data info into nii
        for i in range(data_ply.shape[0]):
            x = int(data_ply[i][0])
            y = int(data_ply[i][1])
            z = int(data_ply[i][2])
            data_nii[x, y, z] = 150

        # save nii
        data_nii = nib.Nifti1Image(data_nii, affine=np.eye(4, 4))
        file_save = name + ".nii.gz"
        nib.save(data_nii, os.path.join(path_pred, case_folder, file_save))


if __name__ == "__main__":
    main()
