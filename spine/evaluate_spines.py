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
The comparison is made at a spine level, finding for each gt spine if it exists on the prediction output.

inputs:
 - ground truth
 - prediction
 - s_min, s_max: size min and size max to discard prediction spines
 - coinc_thr: coincidence threshold to establish if a spine on pred corresponds to a spine in gt
 
outputs:
 - validation metrics (excel)

execution example:
 - python3 evaluate_spines.py --path_run "results/spines/128x128x64_da_medium_300_wdl_sigmoid" --s_min 50 --s_max 45000 --coinc_thr 0.3
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--s_min', default=50,  type=int, help='min spine size')
    parser.add_argument('--s_max', default=45000, type=int,  help='max spine size')
    parser.add_argument('--coinc_thr', default=0.3,  type=float, help='coincidence threshold')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    s_min = parsed_args.s_min           # get min size
    s_max = parsed_args.s_max           # get max size
    coinc_thr = parsed_args.coinc_thr   # get coincidence threshold

    path_pred = os.path.join(path_run, "prediction")

    # initialization
    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision']
    FP = list()
    FN = list()
    TP = list()
    SENS = list()
    PREC = list()
    validation_cases = list()

    dir = listdir(path_pred)

    for case_folder in dir:  # for each case

        # init
        tp_case = 0
        fn_case = 0
        used_list = list()  # already detected spines

        path, val_case = os.path.split(case_folder)

        # load gt and prediction files
        truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        # adapt gt and prediction
        truth.astype(int)

        sp_pred = np.where(prediction == [0])
        bg_pred = np.where(prediction == [-1])
        prediction = np.empty_like(truth)
        prediction[sp_pred] = 255
        prediction[bg_pred] = 0

        # get gt and prediction labels
        label_prediction, num_labels_prediction = label(prediction)
        label_truth, num_labels_truth = label(truth)

        # get gt and predictions spines
        props_pred = regionprops(label_prediction)
        props_truth = regionprops(label_truth)

        # preprocess prediction spines
        for spinePred in range(num_labels_prediction):  # for each spine
            size = props_pred[spinePred].area  # get size
            if size <= s_min or size >= s_max:  # if not in between thresholds
                prediction[props_pred[spinePred].coords[:, 0], props_pred[spinePred].coords[:, 1], props_pred[spinePred].coords[:, 2]] = 0  # delete spine

        # get new prediction labels and spines
        label_prediction, num_labels_prediction = label(prediction)
        props_pred = regionprops(label_prediction)

        for spineGT in range(num_labels_truth):  # for each spine in gt (spineGT)

            # print progression
            prog = (spineGT/num_labels_truth)*100
            print("case: " + case_folder + " - Progreso gt to pred: " + str(round(prog, 1)) + "%")

            # init
            coincide_list_GT = list()
            coincide_list_Pred = list()

            coordsGT = props_truth[spineGT].coords  # get spineGT coords

            for spinePred in range(num_labels_prediction):  # for each spine in prediction (spinePred)

                # init
                counter_sp_coord = 0

                coordsPred = props_pred[spinePred].coords  # get spinePred coords

                for pos in coordsGT:  # for each pixel in SpineGT
                    find = np.where((coordsPred == pos).all(axis=1))  # look if it is in spinePred
                    if find[0].size == 1:  # if it is, count 1
                        counter_sp_coord += 1

                # calculate % of pixels found, respect gt and pred size
                percentageGT = counter_sp_coord / props_truth[spineGT].area
                percentagePred = counter_sp_coord / props_pred[spinePred].area

                # save %
                coincide_list_GT.append(percentageGT)
                coincide_list_Pred.append(percentagePred)

            # delete % from positions of already detected spines
            for ind in used_list:
                coincide_list_GT[ind] = 0
                coincide_list_Pred[ind] = 0

            # get maximum mean score
            coincide_list_mean = [(x+y)/2 for x, y in zip(coincide_list_GT, coincide_list_Pred)]  # scores mean
            max_coinc = max(coincide_list_mean)  # max mean score
            max_index = coincide_list_mean.index(max_coinc)   # max mean score index

            # check if spine is detected
            if max_coinc > coinc_thr:  # if max_coinc is > than coinc_thr
                tp_case = tp_case + 1  # tp + 1
                used_list.append(max_index)  # spine detected
            else:
                fn_case = fn_case + 1  # if not, fn + 1

        fp_case = num_labels_prediction - tp_case  # get fp as the difference between detected spines and tp

        # calculate evaluation metrics
        sensitivity = tp_case / (tp_case + fn_case)
        precision = tp_case / (tp_case + fp_case)

        # save case metrics
        FP.append(fp_case)
        FN.append(fn_case)
        TP.append(tp_case)
        SENS.append(sensitivity)
        PREC.append(precision)
        validation_cases.append(val_case)

    # save spine results on csv
    spine_csv = ({header[0]: FP, header[1]: FN, header[2]: TP, header[3]: SENS, header[4]: PREC})
    df = pd.DataFrame.from_records(spine_csv, index=validation_cases)
    df.to_csv(path_run + "/spine_scores.csv")


if __name__ == "__main__":
    main()
