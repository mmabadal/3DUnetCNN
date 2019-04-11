import numpy as np
import os
from os import listdir
import nibabel as nib
import matplotlib
import argparse
import sys
matplotlib.use('agg')
from skimage import io
from natsort import natsorted

'''
script to create nii files for the data and GT

inputs:
 - refined grayscale tiffs images
 - ground truth tiffs containing the dendrite and spines

outputs:
 - .nii files

execution example:
 - python3 create_nii.py --path_or "../../data/original_grey_chull_tiff" --path_gt "../../data/dendrite_spine_seg_tiff" --path_out "../../data/spines_nii"

'''





def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_or', help='path to the refined grayscale images folder.')
    parser.add_argument('--path_gt', help='path to the unified gt folder.')
    parser.add_argument('--path_out', help='path to the output folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_or = parsed_args.path_or  # get refined grayscale images folder
    path_gt = parsed_args.path_gt  # get unified gt (dendrites+spines) folder folder
    path_out = parsed_args.path_out  # get output folder
    os.mkdir(path_out)

    dir_folders = listdir(path_or)

    for case_folder in dir_folders:

        first = 1
        os.mkdir(os.path.join(path_out, case_folder))

        # SPINE

        dir_slices_or = listdir(os.path.join(path_or, case_folder))
        dir_slices_or = natsorted(dir_slices_or)

        for slice in dir_slices_or:
            # load original files
            spine_slice = io.imread(os.path.join(path_or, case_folder, slice))

            if first == 1:
                spine = spine_slice
                first = 0
            else:
                spine = np.dstack((spine, spine_slice))
        # Save new spine.nii
        spine = nib.Nifti1Image(spine, affine=np.eye(4, 4))
        nib.save(spine, os.path.join(path_out, case_folder, "spine.nii.gz"))

        # TRUTH
        first = 1

        dir_slices_gt = listdir(os.path.join(path_gt, case_folder + "" ))  # case_folder, si tienen sufijo las carpetas de los casos, añadir aqui 
        dir_slices_gt = natsorted(dir_slices_gt)

        for slice in dir_slices_gt:
            
            # load original files
            truth_slice = io.imread(os.path.join(path_gt, case_folder + "", slice, ))  # case_folder, si tienen sufijo las carpetas de los casos, añadir aqui 

            if first == 1:
                truth = truth_slice#[..., 0] # si el gt estan en rgb, desecomentar coger capa 0
                first = 0
            else:
                truth =  np.dstack((truth, truth_slice))#[..., 0] # si el gt estan en rgb, descomentar y añadir coger capa 0 como arriba

        # Save new truth.nii
        truth = nib.Nifti1Image(truth, affine=np.eye(4, 4))
        nib.save(truth, os.path.join(path_out, case_folder, "truth.nii.gz"))



if __name__ == "__main__":
    main()
