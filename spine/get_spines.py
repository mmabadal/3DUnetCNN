import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
import numpy as np
import nibabel as nib
import os
import glob
import argparse
import sys
from os import listdir, path
import matplotlib

'''
script to obtain the spines from ground truth and prediction
inputs:
 - ground truth and prediction of a case
outputs:
 - GT and prediction of all spines from the case
execution example:
 - python3 get_spines.py --path_run "results/128x128x64_da_medium_300_wdl_sigmoid"
'''

def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get path of the predictions
    '''

    path_run = "good/spine_nii"

    
    black_slice = np.zeros((60, 60), dtype=np.uint8)  # aux black slice
    
    x_max = 60 #Max x_size for the desired bounding box
    y_max = 60 #Max y_size for the desired bounding box
    z_stack = 20 #Max z_size for the desired bounding box



    #path_pred = os.path.join(path_run, "prediction")
    path_pred = os.path.join(path_run)

    dir = listdir(path_pred)

    path_pred_spine = os.path.abspath(os.path.join(path_run,"prediction_spine"))
    if path.exists(path_pred_spine) != True:
        os.mkdir(path_pred_spine)

    for case_folder in dir:

        print("Case: " + case_folder)

        case_folder_spine = os.path.abspath(os.path.join(path_pred_spine,case_folder))
        if path.exists(case_folder_spine) != True:
            os.mkdir(case_folder_spine)

        # load gt and spine files
        truth_file = os.path.join(path_pred, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        spines_file = os.path.join(path_pred, case_folder, "spine.nii.gz")
        spines_image = nib.load(spines_file)
        spines = spines_image.get_data()

        # get gt labels
        label_truth, num_labels_truth = label(truth)

        # get gt spines
        props_truth = regionprops(label_truth)

        # get the x,y,z coordinates from the bounding box related to each spine
        for spine in range(num_labels_truth):
            spine_bb = props_truth[spine].bbox
            spine_x=spine_bb[3]-spine_bb[0]
            spine_y=spine_bb[4]-spine_bb[1]
            spine_z=spine_bb[5]-spine_bb[2]

            if (spine_x<=60 and spine_y<=60 and spine_z<=20): 
                spine_folder = os.path.abspath(os.path.join(case_folder_spine,"spine_"+str(spine+1)))
                if path.exists(spine_folder) != True:
                    os.mkdir(spine_folder)
                
                #Obtain BB from GT and fill to get the desired size
                bb = truth[spine_bb[0]:spine_bb[3],spine_bb[1]:spine_bb[4],spine_bb[2]:spine_bb[5]]
                final_bb = np.empty([x_max,y_max,bb.shape[2]], dtype=np.uint8)
                
                for ind in range(bb.shape[2]):
                    b_s = np.zeros((60, 60), dtype=np.uint8)  # aux black slice
                    s = bb[:,:,ind]
                    b_s[0:spine_x,0:spine_y] = s
                    final_bb[:,:,ind] = b_s

                indx = final_bb.shape[2]
                while indx != z_stack:
                    final_bb = np.dstack((final_bb, black_slice))
                    indx += 1

                sp = nib.Nifti1Image(final_bb, affine=np.eye(4, 4))
                nib.save(sp, os.path.join(spine_folder, "truth.nii.gz"))

                #Obtain BB from prediction and fill 'till the desired size
                bb = spines[spine_bb[0]:spine_bb[3],spine_bb[1]:spine_bb[4],spine_bb[2]:spine_bb[5]]
                final_bb = np.empty([x_max,y_max,bb.shape[2]], dtype=np.uint16)

                for ind in range(bb.shape[2]):
                    b_s = np.zeros((60, 60), dtype=np.uint16)  # aux black slice
                    s = bb[:,:,ind]
                    b_s[0:spine_x,0:spine_y] = s
                    final_bb[:,:,ind] = b_s

                indx = final_bb.shape[2]
                while indx != z_stack:
                    final_bb = np.dstack((final_bb, black_slice))
                    indx += 1

                sp = nib.Nifti1Image(final_bb, affine=np.eye(4, 4))
                nib.save(sp, os.path.join(spine_folder, "spine.nii.gz"))


if __name__ == "__main__":
    main()