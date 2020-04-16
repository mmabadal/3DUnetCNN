import numpy as np
import nibabel as nib
import os
import glob
import argparse
import sys
from os import listdir
from scipy import ndimage
import scipy


'''
script to export an image of each slice of nii file

inputs: 
 - nii stack
outputs:
 - images of each slice

execution example:
 - python3 export_slices.py --path_data "../../data/spines_nii" --path_out "../../data/raw/original_grey_tiff_sliced" --origen "spine" --r 150 --g 100 --b 200
 
'''


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', help='path to the input case folders .')
    parser.add_argument('--path_out', help='path to export the images.')
    parser.add_argument('--origen', help='working files name.')
    parser.add_argument('--obj_id', help='working files name.')
    parser.add_argument('--r', help='red')
    parser.add_argument('--g', help='red')
    parser.add_argument('--b', help='red')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_data = parsed_args.path_data  # get data path
    path_out = parsed_args.path_out  # get output path
    origen = parsed_args.origen  # get output path
    obj_id = parsed_args.obj_id
    r = int(parsed_args.r)  # get data path
    g = int(parsed_args.g)  # get data path
    b = int(parsed_args.b)  # get data path

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    name = origen + ".nii.gz"

    dir = listdir(path_data)


    for case_folder in dir:  # for each case


        folder_out = os.path.join(path_out, case_folder, origen + '_tiff')
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        # load nii
        file = os.path.join(path_data, case_folder, name)
        image = nib.load(file)
        data = image.get_data()

        # extract and save slices
        for sl in range(data.shape[2]):

            slice = data[..., sl]
            obj = np.where(slice == [obj_id])

            img = np.zeros([slice.shape[0], slice.shape[1], 3], dtype=np.uint8)  # auxiliary image
            img[obj[0], obj[1], 0] = r  # set red layer
            img[obj[0], obj[1], 1] = g  # set red layer
            img[obj[0], obj[1], 2] = b  # set red layer

            scipy.misc.imsave(folder_out + "/slice_" + str(sl) + ".tiff", img)


if __name__ == "__main__":
    main()
