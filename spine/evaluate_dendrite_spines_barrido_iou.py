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
 - python3 evaluate_dendrite_spines_barrido.py --path_run "results/spines_dendrite/128x128x32_wloss_300ep" --s_min 50 --s_max 45000
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
    FP_d = list()
    FN_d = list()
    TP_d = list()
    SENS_d = list()
    PREC_d = list()
    threshold_global = list()
    validation_cases = list()

    for index in range(0, 10, 1):  # threshold scan

        IoU_thr = index/10  # from 0 to 1 in 0.1 steps

        print("treshold - "+ str(IoU_thr))

        FP_thr = np.array([])
        FN_thr = np.array([])
        TP_thr = np.array([])

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

            print("treshold - "+ str(IoU_thr) + " - case - " + str(idx+1) + "/" + str(len(dir)) + " - " + case_folder)

            '''  DENDRITE  '''

            if index == 0:  # run just the first time

                print("evaluacion dendrita")

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

            print("evaluacion espina")

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
                print("treshold - "+ str(IoU_thr) + " - case - " + str(idx+1) + "/" + str(len(dir)) + " - " + case_folder + " - " + str(round(prog, 1)) + "%")

                # init
                IoU_list = list()

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
                    IoU = counter_sp_coord / (props_truth[spineGT].area + props_pred[spinePred].area - counter_sp_coord)

                    # save %
                    IoU_list.append(IoU)

                # delete % from positions of already detected spines
                for ind in used_list:
                    IoU_list[ind] = 0

                # get maximum mean score
                max_IoU = max(IoU_list)  # max mean score
                max_index = IoU_list.index(max_IoU)  # max mean score index

                # check if spine is detected
                if max_IoU > IoU_thr:  # if max_coinc is > than coinc_thr
                    tp_case = tp_case + 1  # tp + 1
                    used_list.append(max_index)  # spine detected
                else:
                    fn_case = fn_case + 1  # if not, fn + 1

            fp_case = num_labels_prediction - tp_case  # get fp as the difference between detected spines and tp

            # save case metrics
            FP_thr = np.append(FP_thr, fp_case)
            FN_thr = np.append(FN_thr, fn_case)
            TP_thr = np.append(TP_thr, tp_case)

        # save dendrite results on csv
        dendrite_csv = ({header[0]: FP_d, header[1]: FN_d, header[2]: TP_d, header[3]: SENS_d, header[4]: PREC_d})
        df = pd.DataFrame.from_records(dendrite_csv, index=validation_cases)
        df.to_csv(path_run + "/dendrite_scores.csv")


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
        threshold_global.append(IoU_thr)

    # save spine results on csv
    spine_csv = ({header[0]: FP_global, header[1]: FN_global, header[2]: TP_global, header[3]: SENS_global, header[4]: PREC_global})
    df = pd.DataFrame.from_records(spine_csv, index=threshold_global)
    df.to_csv(path_run + "/spine_scores.csv")


if __name__ == "__main__":
    main()
