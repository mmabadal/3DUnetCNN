import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import argparse
from os import listdir
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
import matplotlib

'''
script to evaluate tmerge spine and dendrite results.

inputs:

 - path with:
    * spines predictions
    * dendrites predictions

outputs:
 - spines + dendrites predictions

execution example:
 - python3 merge_results.py --path_spine "results/spine/x" --path_dendrite "results/dendrite/x" --path_out "results/merged/x"
'''


def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_spine', help='path to the spine detection folder.')
    parser.add_argument('--path_dendrite', help='path to the dendrite detection folder.')
    parser.add_argument('--path_out', help='path to store merged predictions.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_spine = parsed_args.path_spine
    path_dendrite = parsed_args.path_dendrite
    path_out =  parsed_args.path_out
    '''

    path_spine = "results/spines/128x128x64_da_medium_300_wdl_sigmoid/"
    path_dendrite = "results/dendrite/128x128x64_da_medium_300_wdl_sigmoid/"
    path_out ="results/merged/128x128x64_da_medium_300_wdl_sigmoid/"

    path_pred_spine = os.path.join(path_spine, "prediction")
    path_pred_dendrite = os.path.join(path_dendrite, "prediction")
    path_pred_out = os.path.join(path_out, "prediction")

    spines = listdir(path_pred_spine)

    for case_folder in spines:  # for each case

        path, val_case = os.path.split(case_folder)
        print("evaluating case" + str(val_case))

        # load spine prediction files
        p_spine = os.path.join(path_pred_spine, case_folder, "prediction.nii.gz")
        nii_spine = nib.load(p_spine)
        spine = nii_spine.get_data()

        # load dendrite prediction files
        p_dendrite = os.path.join(path_pred_dendrite, case_folder, "prediction.nii.gz")
        nii_dendrite = nib.load(p_dendrite)
        dendrite = nii_dendrite.get_data()

        # spine and dendrite coords
        spine_coord = np.where(spine == [0])
        dendrite_coord = np.where(dendrite == [0])

        merged = np.zeros((1024, 1024, 112), dtype=np.uint8)  # create merged matrix

        # fill merged matrix
        merged[dendrite_coord] = 150
        merged[spine_coord] = 255

        # Save merged
        nii_merged = nib.Nifti1Image(merged, affine=np.eye(4, 4))
        nib.save(nii_merged, os.path.join(path_pred_out, case_folder, "prediction.nii.gz"))

if __name__ == "__main__":
    main()




