import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tables
from nibabel.testing import data_path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

def get_whole_spine_mask(data):
    return data == 0


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main():
    header = ['TN', 'FP', 'FN', 'TP', 'Sens.', 'Spec.', 'Precision', 'Accuracy', 'Fall-out', 'Dice_coefficient']

    masking_functions = (get_whole_spine_mask,)
    TN = list()
    FP = list()
    FN = list()
    TP = list()
    Sens = list()
    Spec = list()
    P = list()
    A = list()
    FO = list()
    DC = list()
    validation_cases = list()

    for case_folder in glob.glob("prediction/validation_case*"):
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        path, val_case = os.path.split(case_folder)
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        truth.astype(int)
        ceros = np.where(prediction == [0])
        unos = np.where(prediction == [-1])
        prediction = np.empty_like(truth)
        prediction[ceros] = 255
        prediction[unos] = 0

        cnf_matrix = np.zeros((2, 2))  # auxiliary image

        for slice in range(prediction.shape[2]):

            pred_slice = prediction[..., slice]
            truth_slice = truth[..., slice]

            pred_flat = pred_slice.flatten()
            truth_flat = truth_slice.flatten()

            cnf = confusion_matrix(truth_flat, pred_flat)
            s = cnf.shape

            if s == (1, 1):
                n = pred_flat.size
                valor = pred_flat[0]

                if valor == 255:
                    cnf_matrix[1, 1] = cnf_matrix[1, 1] + n
                elif valor == 0:
                    cnf_matrix[0, 0] = cnf_matrix[0, 0] + n

            else:
                cnf_matrix = cnf + cnf_matrix

        cnf_norm = normalize(cnf_matrix, norm='l1', axis=1)
        tn_norm = cnf_norm[0][0]
        fp_norm = cnf_norm[0][1]
        fn_norm = cnf_norm[1][0]
        tp_norm = cnf_norm[1][1]
        TN.append(tn_norm)
        FP.append(fp_norm)
        FN.append(fn_norm)
        TP.append(tp_norm)

        validation_cases.append(val_case)
        tn = cnf_matrix[0][0]
        fp = cnf_matrix[0][1]
        fn = cnf_matrix[1][0]
        tp = cnf_matrix[1][1]
        Sens.append(tp/(tp+fn))
        Spec.append(tn/(tn+fp))
        P.append(tp/(tp+fp))
        A.append((tp+tn)/(tp+fp+fn+tn))
        FO.append(fp/(fp+tn))
        DC.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    d = ({header[0]: TN, header[1]: FP, header[2]: FN, header[3]: TP, header[4]: Sens, header[5]: Spec,
         header[6]: P, header[7]: A, header[8]: FO, header[9]: DC})

    df = pd.DataFrame.from_records(d, index=validation_cases)

    df.to_csv("./prediction/spine_scores.csv")



    training_df = pd.read_csv("./training.log").set_index('epoch')
    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
