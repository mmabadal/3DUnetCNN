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
script to evaluate the network performance, takes de prediction output and compares it to the ground truth.

inputs:

 - run path with:
    * ground truth
    * prediction
 
outputs:
 - validation metrics (excel)

execution example:
 - python3 evaluate_dendrite2.py --path_run "results/dendrite/128x128x64_da_medium_300_wdl_sigmoid"
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    path_pred = os.path.join(path_run, "prediction")

    # initialization
    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision']
    FP_d = list()
    FN_d = list()
    TP_d = list()
    SENS_d = list()
    PREC_d = list()
    validation_cases = list()

    dir = listdir(path_pred)

    for case_folder in dir:  # for each case

        path, val_case = os.path.split(case_folder)

        # load gt and prediction files
        truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        '''  DENDRITE  '''

        print("dendrite case: " + case_folder)

        # get dendrite coords fot GT and prediction
        den_pred = np.where(prediction != [-1])
        den_gt = np.where(truth != [0])

        # transpose
        den_pred_t = np.transpose(np.asarray(den_pred))
        den_gt_t = np.transpose(np.asarray(den_gt))

        tp_d = 0  # init

        # count number of coincident coords
        for element in den_pred_t:
            find = np.where((den_gt_t == element).all(axis=1))
            if find[0].size == 1:
                tp_d = tp_d + 1

        # calculate evaluation metrics
        fp_d = den_pred_t.shape[0] - tp_d
        fn_d = den_gt_t.shape[0] - tp_d

        # append dendrite metrics
        FP_d.append(fp_d)
        FN_d.append(fn_d)
        TP_d.append(tp_d)
        SENS_d.append(tp_d / (tp_d + fn_d))
        PREC_d.append(tp_d / (tp_d + fp_d))
        validation_cases.append(val_case)

    # save dendrite results on csv
    dendrite_csv = ({header[0]: FP_d, header[1]: FN_d, header[2]: TP_d, header[3]: SENS_d, header[4]: PREC_d})
    df = pd.DataFrame.from_records(dendrite_csv, index=validation_cases)
    df.to_csv(path_run + "/dendrite_scores.csv")


if __name__ == "__main__":
    main()
