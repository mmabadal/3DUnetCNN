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

 - python3 evaluate_spines_nomean.py --path_run "results/spines/128x128x64_da_medium_300_wdl_sigmoid" --v_min 0 --v_max 99999 --coinc_thr 0.5 --print_opt 1 --v_min_sweep 300 --source "spden" --eval "iou"

'''

def get_metrics_from_volumes(TP_v, FP_v, thr, n_gt):

    TP_v_del = [i for i in TP_v if i > thr]
    FP_v_del = [i for i in FP_v if i > thr]

    tp = len(TP_v_del)
    fp = len(FP_v_del)
    fn = n_gt - tp

    if tp == 0:
        sens = 0
        prec = 0
        f1 = 0
    else:
        sens = tp / (tp + fn)
        prec = tp / (tp + fp)
        f1 = (2*(sens*prec)/(sens+prec))


    return sens, prec, f1


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
    parser.add_argument('--coinc_thr', default=0.5,  type=float, help='coincidence threshold')
    parser.add_argument('--print_opt', default=0,  type=int, help='print optins, 0:no print, 1: print all, 2: print together, 3:print separated')
    parser.add_argument('--v_min_sweep', default=0,  type=int, help='v_min_seep options, 0:no sweep, x: sweep from 0 to x ')
    parser.add_argument('--source',  type=str, help='source training classes, options: spine or spden')
    parser.add_argument('--eval', default="iou",  type=str, help='eval options, nomean or iou')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get prediction folder
    v_min = parsed_args.v_min           # get min size
    v_max = parsed_args.v_max           # get max size
    coinc_thr = parsed_args.coinc_thr   # get coincidence threshold
    print_opt = parsed_args.print_opt
    v_min_sweep = parsed_args.v_min_sweep
    source = parsed_args.source
    eval = parsed_args.eval


    path_pred = os.path.join(path_run, "prediction")
    out = str(v_min) + '_' + str(v_max) + '_' + str(coinc_thr) + '_' + str(v_min_sweep) + '_' + eval
    path_out = os.path.join(path_run, "results/", out)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # initialization
    TP_v_all = list()
    FP_v_all = list()
    FP = list()
    FN = list()
    TP = list()
    SENS = list()
    PREC = list()
    F1 = list()
    validation_cases = list()
    SENS_v_all = list()
    PREC_v_all = list()
    F1_v_all = list()

    dir = listdir(path_pred)

    for idx, case_folder in enumerate(dir):  # for each case

        # lists to store volume of TP and FP pred spines
        SCORES = list()
        TP_v = list()
        FP_v = list()

        # lists to store sens and prec calculated from tpv and fpv for each thr sweep
        SENS_v = list()
        PREC_v = list()
        F1_v = list()

        path, val_case = os.path.split(case_folder)

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
                prediction[props_pred[spinePred].coords[:, 0], props_pred[spinePred].coords[:, 1], props_pred[spinePred].coords[:, 2]] = 0  # delete spine

        # get new prediction labels and spines
        label_prediction, num_labels_prediction = label(prediction)
        props_pred = regionprops(label_prediction)

        for spineGT in range(num_labels_truth):  # for each spine in gt (spineGT)

            # print progression
            prog = (spineGT/num_labels_truth)*100
            print("case - " + str(idx + 1) + "/" + str(len(dir)) + " - " + case_folder + '-' + str(round(prog, 1)) + "%")

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
                coincide_list_mean = [(x+y)/2 for x, y in zip(coincide_list_GT, coincide_list_Pred)]  # scores mean
                SCORES.append(coincide_list_mean)

            if eval == "iou":
                SCORES.append(IoU_list)

        tp_case, fp_case, fn_case, used_list = get_cmatrix(SCORES, coinc_thr)

        # calculate evaluation metrics
        sens_case = tp_case / (tp_case + fn_case)
        prec_case = tp_case / (tp_case + fp_case)
        f1_case = (2 * (sens_case * prec_case) / (sens_case + prec_case))

        # save case metrics
        FP.append(fp_case)
        FN.append(fn_case)
        TP.append(tp_case)
        SENS.append(sens_case)
        PREC.append(prec_case)
        F1.append(f1_case)
        validation_cases.append(val_case)

        for i in range(num_labels_prediction):
            if i in used_list:
                TP_v.append(props_pred[i].area)
                TP_v_all.append(props_pred[i].area)
            if i not in used_list:
                FP_v.append(props_pred[i].area)
                FP_v_all.append(props_pred[i].area)

        if v_min_sweep != 0:
            for i in range(v_min_sweep):
                sens, prec, f1 = get_metrics_from_volumes(TP_v, FP_v, i, num_labels_truth)

                SENS_v.append(sens)  # stack sens for each thr of the case
                PREC_v.append(prec)  # stack prec for each thr of the case
                F1_v.append(f1)

            SENS_v_all.append(SENS_v)  # stack sens of all cases for each thr
            PREC_v_all.append(PREC_v)  # stack prec of all cases for each thr
            F1_v_all.append(F1_v)  # stack prec of all cases for each thr



    # save spine results on csv
    header = ['FP', 'FN', 'TP', 'Sens.', 'Precision', 'F1']
    spine_csv = ({header[0]: FP, header[1]: FN, header[2]: TP, header[3]: SENS, header[4]: PREC, header[5]: F1})
    df = pd.DataFrame.from_records(spine_csv, index=validation_cases)
    df.to_csv(path_out + "/spine_scores.ods")

    if v_min_sweep != 0:

        sens_avg = [sum(i)/len(SENS_v_all) for i in zip(*SENS_v_all)]
        prec_avg = [sum(i)/len(PREC_v_all) for i in zip(*PREC_v_all)]
        f1_avg = [sum(i)/len(F1_v_all) for i in zip(*F1_v_all)]

        best_thr = f1_avg.index(max(f1_avg))  # best f1_sweep index = best thr
        best_sens = sens_avg[best_thr]  # sens of best thr
        best_prec = prec_avg[best_thr]  # prec of best thr

        print('best confidence threshold: ' + str(best_thr) + ' with:')
        print('sensitivity: ' + str(best_sens))
        print('precision: ' + str(best_prec))

        plt.plot(sens_avg, 'g', prec_avg, 'r', f1_avg, 'b')
        plt.title('sens and prec vs f1')
        plt.xlabel('V_THR')
        plt.savefig(path_out + "/sweep.pdf")

        # save spine results on csv
        header = ['SENS', 'PREC', 'F1']
        sweep_csv = ({header[0]: sens_avg, header[1]: prec_avg, header[2]: f1_avg})
        df = pd.DataFrame.from_records(sweep_csv, index=range(v_min_sweep))
        df.to_csv(path_out + "/sweep.ods")

    # PLOT
    if print_opt == 1 or print_opt == 2:
    
        fig10 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 5000], color="blue")  # hist(data,bins,range)
        plt.hist(FP_v_all, 300, [0, 5000], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig10.savefig(path_out + '/Histogram_5000.pdf')

        fig20 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 2000], color="blue")  # hist(data,bins,range)
        plt.hist(FP_v_all, 300, [0, 2000], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig20.savefig(path_out + '/Histogram_2000.pdf')
        
        fig30 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 200], color="blue")  # hist(data,bins,range)
        plt.hist(FP_v_all, 300, [0, 200], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig30.savefig(path_out + '/Histogram_200.pdf')

    if print_opt == 1 or print_opt == 3:

        fig10 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 5000], color="blue")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig10.savefig(path_out + '/Histogram_tp_5000.pdf')

        fig11 = plt.figure()
        plt.hist(FP_v_all, 300, [0, 5000], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig11.savefig(path_out + '/Histogram_fp_5000.pdf')

        fig20 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 2000], color="blue")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig20.savefig(path_out + '/Histogram_tp_2000.pdf')

        fig21 = plt.figure()
        plt.hist(FP_v_all, 300, [0, 2000], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig21.savefig(path_out + '/Histogram_fp_2000.pdf')

        fig30 = plt.figure()
        plt.hist(TP_v_all, 300, [0, 200], color="blue")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig30.savefig(path_out + '/Histogram_tp_200.pdf')

        fig31 = plt.figure()
        plt.hist(FP_v_all, 300, [0, 200], color="orange")  # hist(data,bins,range)
        plt.minorticks_on()
        plt.xlabel('spine area')
        plt.ylabel('Number of spines')
        fig31.savefig(path_out + '/Histogram_fp_200.pdf')


if __name__ == "__main__":
    main()
