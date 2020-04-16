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
script to create fake truth nii

inputs:
 - pred nii

outputs:
 - false truth nii

execution example:
 - python3 fake_truth.py --path "data/"

'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the refined grayscale images folder.')

    parsed_args = parser.parse_args(sys.argv[1:])
    path = parsed_args.path
    dir = listdir(path)


    for case_folder in dir:  # for each case

        path_pred, case = os.path.split(case_folder)
        pred_file = os.path.join(path, path_pred, case_folder, "spine.nii.gz")
        pred_nii = nib.load(pred_file)
        nib.save(pred_nii, os.path.join(path, case_folder, "truth.nii.gz"))
        #os.remove(pred_file)


if __name__ == "__main__":
    main()
