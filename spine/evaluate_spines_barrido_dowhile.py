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


def get_cmatrix(SCORES, thr):

    n_gt = len(SCORES)
    n_pred = len(SCORES[0])

    eval_list = list(range(n_gt))
    eval_aux = list()
    pairs = np.full([2,n_gt], -1)
    # lista/matrix de parejas

    while eval_list:
        for i in eval_list:
            ok = False
            gt_scores = SCORES[i]
            max_score = max(gt_scores)
            while ok == False:
                if max_score > thr:
                    max_index = gt_scores.index(max_score)
                    if max_index not in pairs[0,:]:
                        pairs[0,i] = max_index
                        pairs[1,i] = max_score
                        ok = True
                    else:
                        a = np.where(pairs[0,:] == max_index)[0][0]
                        if pairs[1, a] < max_score:
                            pairs[0, i] = max_index
                            pairs[1, i] = max_score
                            pairs[0, a] = -1
                            pairs[1, a] = -1
                            eval_aux.append(a)
                            ok = True
                        if pairs[1, a] > max_score:
                            gt_scores[max_index] = 0
                            max_score = max(gt_scores)
                        else:
                            ok = True
                else:
                    ok = True

        eval_list = eval_aux
        eval_aux = list()

    used_list = list(pairs[0, np.where(pairs[0, :] >= 0)[0]])
    tp = len(used_list)
    fp = n_pred - tp
    fn = n_gt - tp

    return tp, fp, fn, used_list


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--v_min', default=10,  type=int, help='min spine volume')
    parser.add_argument('--v_max', default=99999, type=int,  help='max spine volume')
    parser.add_argument('--source',  type=str, help='source training classes, options: spine or spden')
    parser.add_argument('--eval', default="nomean",  type=str, help='eval options, nomean or iou')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    v_min = parsed_args.v_min           # get min size
    v_max = parsed_args.v_max           # get max size
    source = parsed_args.source
    eval = parsed_args.eval

    path_pred = os.path.join(path_run, "prediction")
    out = str(v_min) + '_' + eval + '_sweep'
    path_out = os.path.join(path_run, "results/", out)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # initialization
    FP_global = list()
    FN_global = list()
    TP_global = list()
    SENS_global = list()
    PREC_global = list()
    F1_global = list()


    threshold_global = list()

    for index in range(0, 10, 1):  # threshold scan

        coinc_thr = index / 10  # from 0 to 1 in 0.1 steps

        print("treshold - " + str(coinc_thr))

        FP_thr = np.array([])
        FN_thr = np.array([])
        TP_thr = np.array([])
        PREC_thr = np.array([])
        SENS_thr = np.array([])
        F1_thr = np.array([])


        dir = listdir(path_pred)

        for idx, case_folder in enumerate(dir):  # for each case

            print("treshold - " + str(coinc_thr) + " - case - " + str(idx + 1) + "/" + str(len(dir)) + " - " + case_folder)

            SCORES = list()

            # load gt and prediction files
            truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
            truth_image = nib.load(truth_file)
            truth = truth_image.get_data()

            prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
            prediction_image = nib.load(prediction_file)
            prediction = prediction_image.get_data()

            if source == "spine":
                # adapt gt and prediction
                truth.astype(int)
                sp_pred = np.where(prediction == [0])
                bg_pred = np.where(prediction == [-1])
                prediction = np.empty_like(truth)
                prediction[sp_pred] = 255
                prediction[bg_pred] = 0

            if source == "spden":
                den_pred = np.where(prediction == [150])
                den_gt = np.where(truth == [150])
                prediction[den_pred[0], den_pred[1], den_pred[2]] = 0
                truth[den_gt[0], den_gt[1], den_gt[2]] = 0

            # get gt and prediction labels
            label_prediction, num_labels_prediction = label(prediction)
            label_truth, num_labels_truth = label(truth)

            # get gt and predictions spines
            props_pred = regionprops(label_prediction)
            props_truth = regionprops(label_truth)

            # preprocess prediction spines
            for spinePred in range(num_labels_prediction):  # for each spine
                size = props_pred[spinePred].area  # get size
                if size <= v_min or size >= v_max:  # if not in between thresholds
                    prediction[props_pred[spinePred].coords[:, 0], props_pred[spinePred].coords[:, 1], props_pred[spinePred].coords[:,2]] = 0  # delete spine

            # get new prediction labels and spines
            label_prediction, num_labels_prediction = label(prediction)
            props_pred = regionprops(label_prediction)

            for spineGT in range(num_labels_truth):  # for each spine in gt (spineGT)

                # print progression
                prog = (spineGT / num_labels_truth) * 100
                print("treshold - " + str(coinc_thr) + " - case - " + str(idx + 1) + "/" + str(
                    len(dir)) + " - " + case_folder + " - " + str(round(prog, 1)) + "%")

                # init
                coincide_list_GT = list()
                coincide_list_Pred = list()
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

                    if eval == "nomean":
                        # calculate % of pixels found, respect gt and pred size
                        percentageGT = counter_sp_coord / props_truth[spineGT].area
                        percentagePred = counter_sp_coord / props_pred[spinePred].area
                        # save %
                        coincide_list_GT.append(percentageGT)
                        coincide_list_Pred.append(percentagePred)

                    if eval == "iou":
                        # calculate % of pixels found, respect gt and pred size
                        IoU = counter_sp_coord / (props_truth[spineGT].area + props_pred[spinePred].area - counter_sp_coord)
                        # save %
                        IoU_list.append(IoU)

                if eval == "nomean":

                    for i in range(len(coincide_list_GT)):
                        if coincide_list_GT[i] < coinc_thr or coincide_list_Pred[i] < coinc_thr:
                            coincide_list_GT[i] = 0
                            coincide_list_Pred[i] = 0
                    # get maximum mean score
                    coincide_list_mean = [(x + y) / 2 for x, y in
                                          zip(coincide_list_GT, coincide_list_Pred)]  # scores mean
                    SCORES.append(coincide_list_mean)

                if eval == "iou":
                    SCORES.append(IoU_list)

            tp_case, fp_case, fn_case, used_list = get_cmatrix(SCORES, coinc_thr)

            prec_case = tp_case/(tp_case+fp_case)
            sens_case = tp_case/(tp_case+fn_case)
            f1_case = (2 * (sens_case * prec_case) / (sens_case + prec_case))

            # save case metrics
            FP_thr = np.append(FP_thr, fp_case)
            FN_thr = np.append(FN_thr, fn_case)
            TP_thr = np.append(TP_thr, tp_case)
            PREC_thr = np.append(PREC_thr, prec_case)
            SENS_thr = np.append(SENS_thr, sens_case)
            F1_thr = np.append(F1_thr, f1_case)


        # calculate and save threshold metrics
        FP_sum = np.sum(FP_thr)
        FN_sum = np.sum(FN_thr)
        TP_sum = np.sum(TP_thr)
        PREC_avg = np.mean(PREC_thr)
        SENS_avg = np.mean(SENS_thr)
        F1_avg = np.mean(F1_thr)

        FP_global.append(FP_sum)
        FN_global.append(FN_sum)
        TP_global.append(TP_sum)
        PREC_global.append(PREC_avg)
        SENS_global.append(SENS_avg)
        F1_global.append(F1_avg)

        threshold_global.append(coinc_thr)

    # save spine results on csv
    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision', 'F1']
    spine_csv = ({header[0]: FP_global, header[1]: FN_global, header[2]: TP_global, header[3]: SENS_global, header[4]: PREC_global, header[5]: F1_global})
    df = pd.DataFrame.from_records(spine_csv, index=threshold_global)
    df.to_csv(path_out + "/spine_scores_barrido.csv")


if __name__ == "__main__":
    main()
