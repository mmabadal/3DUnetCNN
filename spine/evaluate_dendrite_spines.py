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

For the spines, the comparison is made at a instance level, finding for each gt spine if it exists (coincidence > threshold) on the 
prediction output.The evaluation is computed at a selected coincidence threshold.

inputs:
 - ground truth
 - prediction
 - s_min, s_max: size min and size max to discard prediction spines
 
outputs:
 - validation metrics (excel)

execution example:
 - python3 evaluate_dendrite_spines.py --path_run "results/spines_dendrite/128x128x64_da_medium_300_dl_sigmoid" --s_min 50 --s_max 45000 --coinc_thr 0.0
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
    FP_s = list()
    FN_s = list()
    TP_s = list()
    SENS_s = list()
    PREC_s = list()
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

        # append dendrite metrics
        FP_d.append(fp_d)
        FN_d.append(fn_d)
        TP_d.append(tp_d)
        SENS_d.append(tp_d / (tp_d + fn_d))
        PREC_d.append(tp_d / (tp_d + fp_d))
        validation_cases.append(val_case)



        '''  SPINE  '''

        # init
        tp_case = 0
        fn_case = 0
        used_list = list()  # already detected spines

        # delete dendrite from GT and prediction
        prediction[den_pred[0], den_pred[1], den_pred[2]] = 0
        truth[den_gt[0], den_gt[1], den_gt[2]] = 0

        # get gt and prediction labels
        label_prediction, num_labels_prediction = label(prediction)
        label_truth, num_labels_truth = label(truth)

        # get gt and predictions spines
        props_pred = regionprops(label_prediction)  # get prediction blobs
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
        FP_s.append(fp_case)
        FN_s.append(fn_case)
        TP_s.append(tp_case)
        SENS_s.append(sensitivity)
        PREC_s.append(precision)


    # save dendrite results on csv
    dendrite_csv = ({header[0]: FP_d, header[1]: FN_d, header[2]: TP_d, header[3]: SENS_d, header[4]: PREC_d})
    df = pd.DataFrame.from_records(dendrite_csv, index=validation_cases)
    df.to_csv(path_run + "/dendrite_scores.csv")

    # save spine results on csv
    spine_csv = ({header[0]: FP_s, header[1]: FN_s, header[2]: TP_s, header[3]: SENS_s, header[4]: PREC_s})
    df = pd.DataFrame.from_records(spine_csv, index=validation_cases)
    df.to_csv(path_run + "/spine_scores.csv")


if __name__ == "__main__":
    main()
