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
script delete spines outside a size thresholds

inputs:

 - spine prediction
 - s_min, s_max: size min and size max to discard prediction spines

 
outputs:
 - validation metrics (excel)

execution example:
 - python3 delete_spines.py --path_run "results/spines/128x128x64_da_medium_300_wdl_sigmoid" --s_min 50 --s_max 45000
 
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--s_min', default=50,  type=int, help='min spine size')
    parser.add_argument('--s_max', default=45000, type=int,  help='max spine size')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    s_min = parsed_args.s_min           # get min size
    s_max = parsed_args.s_max           # get max size

    path_pred = os.path.join(path_run, "prediction")

    dir = listdir(path_pred)

    for case_folder in dir:  # for each case

        print("deleting spines from case: " + case_folder)
        # load prediction files

        prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
        prediction_nii = nib.load(prediction_file)
        prediction = prediction_nii.get_data()

        # adapt prediction
        sp_pred = np.where(prediction == [0])
        bg_pred = np.where(prediction == [-1])
        prediction_int = np.empty_like(prediction, dtype=np.uint8)

        prediction_int[sp_pred] = 255
        prediction_int[bg_pred] = 0

        # get prediction labels and spines
        label_prediction, num_labels_prediction = label(prediction_int)
        props_pred = regionprops(label_prediction)

        # preprocess prediction spines
        for spinePred in range(num_labels_prediction):  # for each spine
            size = props_pred[spinePred].area  # get size
            if size <= s_min or size >= s_max:  # if outside thresholds
                prediction_int[props_pred[spinePred].coords[:, 0], props_pred[spinePred].coords[:, 1], props_pred[spinePred].coords[:, 2]] = 0  # delete spine

        sp_pred = np.where(prediction_int == [255])
        bg_pred = np.where(prediction_int == [0])
        prediction[sp_pred] = 0
        prediction[bg_pred] = -1

        # Save new spine.nii
        prediction_nii = nib.Nifti1Image(prediction, affine=np.eye(4, 4))
        nib.save(prediction_nii, os.path.join(path_pred, case_folder, "prediction_croped_" + str(s_min) + "_" + str(s_max)  + ".nii.gz"))


if __name__ == "__main__":
    main()
