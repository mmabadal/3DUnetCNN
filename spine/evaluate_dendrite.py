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
 - python3 evaluate_dendrite.py --path_run "results/dendrite/128x128x64_da_medium_300_wdl_sigmoid" --source "spden"
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--source',  type=str, help='source training classes, options: spine or spden')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    path_pred = os.path.join(path_run, "prediction")
    source = parsed_args.source


    # initialization
    FP_list = list()
    FN_list = list()
    TP_list = list()
    SENS_list = list()
    PREC_list = list()
    F1_list = list()
    IoU_list = list()
    validation_cases = list()

    dir = listdir(path_pred)

    for idx, case_folder in enumerate(dir):  # for each case

        path, val_case = os.path.split(case_folder)

        # load gt and prediction files
        truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        '''  DENDRITE  '''
        print("dendrite case - " + str(idx + 1) + "/" + str(len(dir)) + ' - ' + case_folder)

        # get dendrite coords fot GT and prediction
        if source == "dendrite":
            den_pred = np.where(prediction == [0])
            den_gt = np.where(truth == [255])

        if source == "spden":
            den_pred = np.where(prediction == [150])
            den_gt = np.where(truth == [150])

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
        prec = tp_d / (tp_d + fp_d)
        sens = tp_d / (tp_d + fn_d)
        f1 = (2*(prec*sens))/(prec+sens)
        IoU = tp_d/(tp_d+fn_d+fp_d)

        # append dendrite metrics
        FP_list.append(fp_d)
        FN_list.append(fn_d)
        TP_list.append(tp_d)
        SENS_list.append(sens)
        PREC_list.append(prec)
        F1_list.append(f1)
        IoU_list.append(IoU)

        validation_cases.append(val_case)

    # save dendrite results on csv
    path_out = os.path.join(path_run, "results/")
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision', 'F1', 'IoU']
    dendrite_csv = ({header[0]: FP_list, header[1]: FN_list, header[2]: TP_list, header[3]: SENS_list, header[4]: PREC_list, header[5]: F1_list, header[6]: IoU_list})
    df = pd.DataFrame.from_records(dendrite_csv, index=validation_cases)
    df.to_csv(path_out + "/dendrite_scores.ods")


if __name__ == "__main__":
    main()
