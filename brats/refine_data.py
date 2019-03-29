import numpy as np
import os
from os import listdir
import nibabel as nib
import matplotlib
matplotlib.use('agg')
import argparse
import sys
from skimage import io
from natsort import natsorted
from scipy import ndimage
import scipy
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import convex_hull_image


'''
script to refine the data by deleting the dendrites that are not segmented in the GT

inputs:
 - original grayscale tiffs images
 - ground truth tiffs containing the dendrite and spines

outputs:
 -refined data (just the principal dendrite)

execution example:
 - python3 refine_data.py --path_or "../../data/raw/original_grey_tiff" --path_gt "../../data/dendrite_spine_seg_tiff" --path_out "../../data/refine_data_chull"

'''

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_or', help='path to the original grayscale images folder.')
    parser.add_argument('--path_gt', help='path to the unified gt (dendrites+spines) folder.')
    parser.add_argument('--path_out', help='path to the output folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_or = parsed_args.path_or  # get original grayscale images folder
    path_gt = parsed_args.path_gt  # get unified gt (dendrites+spines) folder folder
    path_out = parsed_args.path_out  # get output folder
    os.mkdir(path_out)

    dir_folders = listdir(path_or)

    for case_folder in dir_folders:

        os.mkdir(os.path.join(path_out, case_folder))

        dir_slices_or = listdir(os.path.join(path_or, case_folder))
        dir_slices_or = natsorted(dir_slices_or)

        dir_slices_gt = listdir(os.path.join(path_gt, case_folder))
        dir_slices_gt = natsorted(dir_slices_gt)

        for slice in dir_slices_or:

            indx_slice = dir_slices_or.index(slice)

            # load original files
            or_slice = io.imread(os.path.join(path_or, case_folder, slice))
            gt_slice = io.imread(os.path.join(path_gt, case_folder, dir_slices_gt[indx_slice]))

            pos_object = np.where(gt_slice != [0])
            #print(len(pos_object[0]))
            if len(pos_object[0]) != 0 :
                for pos in range(len(pos_object[0])):
                    gt_slice[pos_object[0][pos], pos_object[1][pos]] = 1

                slice_chull = convex_hull_image(gt_slice)

                slice_chull = slice_chull*1
                slice_dil = dilation(slice_chull, disk(30))

                pos_object_2 = np.where(slice_dil == [0])

                for pos in range(len(pos_object_2[0])):
                    or_slice[pos_object_2[0][pos], pos_object_2[1][pos]] = 0
            else:
                or_slice = np.zeros((1024, 1024), dtype=np.uint8)

            scipy.misc.imsave(path_out + "/" + case_folder + "/" + str(dir_slices_or[indx_slice]), or_slice)


if __name__ == "__main__":
    main()

