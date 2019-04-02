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

The comparison is made at a spine level, finding for each gt spine if it exists (coincidence > threshold) on the 
prediction output.

The evaluation is computed at diverse coincidence threshold values, from 0 to 1 in 0.1 steps, in order to determine 
which one offers the best trade off between tp, fn and fp.

inputs:
 - ground truth
 - prediction
 - s_min, s_max: size min and size max to discard prediction spines
 
outputs:
 - validation metrics (excel)

execution example:
 - python3 evaluate_spines_barrido.py --path_run "results/spines/128x128x32_wloss_300ep" --s_min 50 --s_max 45000
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the prediction folder.')
    parser.add_argument('--s_min', default=50, type=int, help='min spine size')
    parser.add_argument('--s_max', default=45000, type=int, help='max spine size')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    s_min = parsed_args.s_min           # get min size
    s_max = parsed_args.s_max           # get max size

    path_pred = os.path.join(path_run, "prediction")

    # initialization
    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision']
    FP_global = list()
    FN_global = list()
    TP_global = list()
    SENS_global = list()
    PREC_global = list()
    threshold_global = list()

    for index in range(0, 10, 1):  # threshold scan

        coinc_thr = index/10  # from 0 to 1 in 0.1 steps

        FP_thr = np.array([])
        FN_thr = np.array([])
        TP_thr = np.array([])

        dir = listdir(path_pred)

        for case_folder in dir:  # for each case

            # init
            tp_case = 0
            fn_case = 0
            used_list = list()  # already detected spines

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
                coincide_list_mean = [(x + y) / 2 for x, y in zip(coincide_list_GT, coincide_list_Pred)]  # scores mean
                max_coinc = max(coincide_list_mean)  # max mean score
                max_index = coincide_list_mean.index(max_coinc)  # max mean score index

                # check if spine is detected
                if max_coinc > coinc_thr:  # if max_coinc is > than coinc_thr
                    tp_case = tp_case + 1  # tp + 1
                    used_list.append(max_index)  # spine detected
                else:
                    fn_case = fn_case + 1  # if not, fn + 1

            fp_case = num_labels_prediction - tp_case  # get fp as the difference between detected spines and tp

            # save case metrics
            FP_thr = np.append(FP_thr, fp_case)
            FN_thr = np.append(FN_thr, fn_case)
            TP_thr = np.append(TP_thr, tp_case)

        # calculate and save threshold metrics
        FP_sum = np.sum(FP_thr)
        FN_sum = np.sum(FN_thr)
        TP_sum = np.sum(TP_thr)

        SENS_thr = TP_sum / (TP_sum + FN_sum)
        PREC_thr = TP_sum / (TP_sum + FP_sum)

        FP_global.append(FP_sum)
        FN_global.append(FN_sum)
        TP_global.append(TP_sum)
        SENS_global.append(SENS_thr)
        PREC_global.append(PREC_thr)
        threshold_global.append(coinc_thr)

    # save spine results on csv
    spine_csv = ({header[0]: FP_global, header[1]: FN_global, header[2]: TP_global, header[3]: SENS_global, header[4]: PREC_global})
    df = pd.DataFrame.from_records(spine_csv, index=threshold_global)
    df.to_csv(path_run + "/spine_scores.csv")

if __name__ == "__main__":
    main()
