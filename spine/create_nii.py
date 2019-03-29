import numpy as np
import os
from os import listdir
import nibabel as nib
import matplotlib
matplotlib.use('agg')
from skimage import io


def main():

    path_in_or = "/disk/lidia/data/raw/original_grey_chull/"
    path_in_gt = "/disk/lidia/data/raw/spine_seg_tiff/"
    dir_folders = listdir(path_in_or)
    path_out = "/disk/lidia/data/"
    folder_out = "original_nii"
    spine = []
    extend = 1
    add_dendrite = 1

    for case_folder in dir_folders:

        os.mkdir(os.path.join(path_out, folder_out))

        dir_slices_or = listdir(os.path.join(path_in_or, case_folder))
        dir_slices_or = natsorted(dir_slices_or)
        dir_slices_gt = listdir(os.path.join(path_in_gt, case_folder))
        dir_slices_gt = natsorted(dir_slices_gt)

        for slice in dir_slices_or:

            # load original files
            spine_slice = io.imread(os.path.join(path_in_or, case_folder, slice))
            spine = np.dstack((spine, spine_slice))
            truth_slice = io.imread(os.path.join(path_in_gy, case_folder, slice))
            truth = np.dstack((spine, truth_slice))

        # Save new spine.nii
        spine = nib.Nifti1Image(spine, affine=np.eye(4, 4))
        nib.save(spine, os.path.join(path_out, case_folder, "spine.nii.gz"))

        # Save new truth.nii
        truth = nib.Nifti1Image(truth, affine=np.eye(4, 4))
        nib.save(truth, os.path.join(path_out, case_folder, "truth.nii.gz"))



if __name__ == "__main__":
    main()
