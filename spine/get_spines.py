from scipy.ndimage import label
from skimage.measure import regionprops
import numpy as np
import nibabel as nib
import os
import argparse
import sys
from os import listdir, path

'''
script to obtain spines from ground truth and grayscale 
inputs:
 - ground truth and grayscale of a case| path to save the results | x,y and z for the wanted bounding box
outputs:
 - ground truth and grayscale of each spine from the case 
execution example:
 - python3 get_spines.py --path_data "/disk/3d_unet/data/spine_nii" --path_out "/disk/3d_unet/data/spine_nii_output" --x 60 --y 60 --z 20
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', help='path to the data folder.')
    parser.add_argument('--path_out', help='path to export the output case folders.')
    parser.add_argument('--x', default="60", help='x size for resulting BB')
    parser.add_argument('--y', default="60", help='y size for resulting BB')
    parser.add_argument('--z', default="60", help='z size for resulting BB')

    parsed_args = parser.parse_args(sys.argv[1:])
    path_data = parsed_args.path_data  # get path of the GT and grayscale
    path_out = parsed_args.path_out
    x_size= int(parsed_args.x) 
    y_size = int(parsed_args.y) 
    z_size = int(parsed_args.z) 
  
    black_slice = np.zeros((x_size, y_size), dtype=np.uint8)  # aux black slice

    if not path.exists(path_out):
        os.mkdir(path_out)
    
    dir = listdir(path_data)

    for case_folder in dir:

        print("Case: " + case_folder)

        # load gt and spine files
        truth_file = os.path.join(path_data, case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()

        spines_file = os.path.join(path_data, case_folder, "spine.nii.gz")
        spines_image = nib.load(spines_file)
        spines = spines_image.get_data()

        # get gt labels
        label_truth, num_labels_truth = label(truth)

        # get gt spines
        props_truth = regionprops(label_truth)

        # get the x,y,z coordinates from the bounding box related to each spine
        for spine in range(num_labels_truth):
            spine_bb = props_truth[spine].bbox
            spine_bb_x=spine_bb[3]-spine_bb[0]
            spine_bb_y=spine_bb[4]-spine_bb[1]
            spine_bb_z=spine_bb[5]-spine_bb[2]

            if (spine_bb_x<=x_size and spine_bb_y<=y_size and spine_bb_z<=z_size): 
                
                case_folder_spine = os.path.join(path_out,case_folder+"_spine"+str(spine+1))
                if not path.exists(case_folder_spine):
                    os.mkdir(case_folder_spine)
                
                #Obtain BB from GT and fill to get the desired size
                bb = truth[spine_bb[0]:spine_bb[3],spine_bb[1]:spine_bb[4],spine_bb[2]:spine_bb[5]]
                final_bb = np.empty([x_size,y_size,bb.shape[2]], dtype=np.uint8)
                
                for ind in range(bb.shape[2]):
                    b_s = np.zeros((x_size, y_size), dtype=np.uint8)  # aux black slice
                    s = bb[:,:,ind]
                    b_s[0:spine_bb_x,0:spine_bb_y] = s
                    final_bb[:,:,ind] = b_s

                indx = final_bb.shape[2]
                while indx != z_size:
                    final_bb = np.dstack((final_bb, black_slice))
                    indx += 1

                sp = nib.Nifti1Image(final_bb, affine=np.eye(4, 4))
                nib.save(sp, os.path.join(case_folder_spine, "truth.nii.gz"))

                #Obtain BB from prediction and fill 'till the desired size
                bb = spines[spine_bb[0]:spine_bb[3],spine_bb[1]:spine_bb[4],spine_bb[2]:spine_bb[5]]
                final_bb = np.empty([x_size,y_size,bb.shape[2]], dtype=np.uint16)

                for ind in range(bb.shape[2]):
                    b_s = np.zeros((x_size, y_size), dtype=np.uint16)  # aux black slice
                    slice_bb = bb[:,:,ind]
                    b_s[0:spine_bb_x,0:spine_bb_y] = slice_bb
                    final_bb[:,:,ind] = b_s

                indx = final_bb.shape[2]
                while indx != z_size:
                    final_bb = np.dstack((final_bb, black_slice))
                    indx += 1

                sp = nib.Nifti1Image(final_bb, affine=np.eye(4, 4))
                nib.save(sp, os.path.join(case_folder_spine, "spine.nii.gz"))


if __name__ == "__main__":
    main()
