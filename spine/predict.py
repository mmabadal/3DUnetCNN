import os
import numpy as np
import os
import argparse
import sys
from os import listdir
import nibabel as nib
from os import rename
from skimage import io
from train import config
from unet3d.prediction import run_validation_cases
import matplotlib
matplotlib.use('agg')


'''
script to perform inference over a set of cases, performing the class prediction

inputs:
 - grey scale data .nii
 
outputs:
 - grey scale data .nii
 - class prediction .nii
 - ground truth .nii
 
execution example:
 - python3 predict.py --path_run "results/spines/128x128x64_da_medium_300_wdl_sigmoid"
 
'''


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run   # get run folder

    #path_pred = os.path.join(path_run, "prediction")

    #path_pred = os.path.abspath(path_run + "/prediction")

    path_pred = os.path.abspath("prediction")

    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=False,
                         output_dir=path_pred)

    '''the network outputs the nii stacks in a different resolution than the originals introduced, in order to make
    them equal, these are loaded into python and saved again. '''

    dir_pred = listdir(path_pred)

    for case_folder in dir_pred:

        # load predicted files
        spine_file = os.path.join(path_pred, case_folder, "data_spine.nii.gz")
        truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
        prediction_file = os.path.join(path_pred, case_folder, "prediction.nii.gz")
        spine_image = nib.load(spine_file)
        truth_image = nib.load(truth_file)
        prediction_image = nib.load(prediction_file)
        spine = spine_image.get_data()
        truth = truth_image.get_data()
        prediction = prediction_image.get_data()

        # save predicted files
        spine = nib.Nifti1Image(spine, affine=np.eye(4, 4))
        nib.save(spine, spine_file)
        truth = nib.Nifti1Image(truth, affine=np.eye(4, 4))
        nib.save(truth, truth_file)
        prediction = nib.Nifti1Image(prediction, affine=np.eye(4, 4))
        nib.save(prediction, prediction_file)

    # rename predicted files to match the original ones

    path_original = "data/"  # get path of original data
    dir_original = listdir(path_original)  # list of original cases
    for case_pred in dir_pred:  # for each predicted case
        name1, name2, number = case_pred.split("_")  # get number
        rename(path_pred + "/" + case_pred, path_pred + "/" + dir_original[int(number)])  # rename to match original case

if __name__ == "__main__":
    main()






