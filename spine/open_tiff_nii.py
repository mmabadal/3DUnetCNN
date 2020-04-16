import numpy as np
import os
from skimage import io
import nibabel as nib


path_tiff_1 = "../data/good/raw/original_grey_tiff/if6.1.1-1enero/if6 1 1-1enero_Z027.tif"
path_tiff_2 = "../data/good/raw/original_grey_tiff/m16.2.10-2/m16 2 10-2_Z027.tif"
path_nii_1 = "../data/good/dendrite_spine_nii_nochull_python/if6.1.1-1enero/spine.nii.gz"
path_nii_2 = "../data/good/dendrite_spine_nii_nochull_python/m16.2.10-2/spine.nii.gz"
path_nii_extended_1 = "../data/good/dendrite_spine_nii_nochull_extended_python/if6.1.1-1enero/spine.nii.gz"
path_nii_extended_2 = "../data/good/dendrite_spine_nii_nochull_extended_python/m16.2.10-2/spine.nii.gz"

spine_slice_1 = io.imread(path_tiff_1)
spine_slice_1 = np.flipud(spine_slice_1)
spine_slice_1 = np.rot90(spine_slice_1)
spine_slice_1 = np.rot90(spine_slice_1)
spine_slice_1 = np.rot90(spine_slice_1)

spine_slice_2 = io.imread(path_tiff_2)
spine_slice_2 = np.flipud(spine_slice_2)
spine_slice_2 = np.rot90(spine_slice_2)
spine_slice_2 = np.rot90(spine_slice_2)
spine_slice_2 = np.rot90(spine_slice_2)

data_nii_1 = nib.load(path_nii_1)
nii_1 = data_nii_1.get_data()

data_nii_2 = nib.load(path_nii_2)
nii_2 = data_nii_2.get_data()

data_nii_extended_1 = nib.load(path_nii_extended_1)
nii_extended_1 = data_nii_extended_1.get_data()

data_nii_extended_2 = nib.load(path_nii_extended_2)
nii_extended_2 = data_nii_extended_2.get_data()


path_nii_2_norm = "../data/good/a/nii_2_norm.nii.gz"
path_nii_2_norm_chull = "../data/good/a/nii_2_norm_chull.nii.gz"

data_nii_2_norm = nib.load(path_nii_2_norm)
nii_2_norm = data_nii_2_norm.get_data()
norm_max = np.max(nii_2_norm)
norm_min = np.min(nii_2_norm)

data_nii_2_norm_chull = nib.load(path_nii_2_norm_chull)
nii_2_norm_chull = data_nii_2_norm_chull.get_data()
norm_chull_max = np.max(nii_2_norm_chull)
norm_chull_min = np.min(nii_2_norm_chull)




nii_2_norm = nii_2
mean = nii_2_norm.mean(axis=(0, 1, 2))
std = nii_2_norm.std(axis=(0, 1, 2))
a = nii_2_norm - mean
b = a/std

path_out = "../data/good/a/nii_2_norm.nii.gz"
data_out = nib.Nifti1Image(b, affine=None)
nib.save(data_out, path_out)

z = 1

