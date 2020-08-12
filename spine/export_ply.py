import os
import glob
import numpy as np
import argparse
from os import listdir
import sys
import nibabel as nib

'''
script to export a nii file as ply

inputs:
 - nii file
outputs:
 - ply file

execution example:

 - python3 export_ply.py --path_run "results/merged/x/" --rescale 1 --name_in "prediction" --class2exp 0

'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--name_in', help='working file name')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get prediction folder
    name_in = parsed_args.name_in  # get type


    file = name_in + ".nii.gz"

    path_pred = os.path.join(path_run, "prediction")

    dir = listdir(path_pred)


    for case_folder in dir:

        # load nii file
        file_path = os.path.join(path_pred, case_folder, file)
        prediction = nib.load("/home/miguel/Desktop/3DUnetCNN/spine/RUNS/test_5_128_64/prediction/Cas002/prediction.nii.gz")
        prediction = prediction.get_data()

        pred = np.where(prediction > [0.70])  # get detected index
        name_out = "bone"

        # get detected coords

        x = pred[0]
        y = pred[1]
        z = pred[2]
        coords = np.zeros((x.shape[0], 3), dtype=int)  # auxiliary image


        z = 0.25 * z


        # append information
        for coord in range(x.shape[0]):
            coords[coord, 0] = int(x[coord])
            coords[coord, 1] = int(y[coord])
            coords[coord, 2] = int(z[coord])

        # create ply
        out_path = os.path.join(path_pred, case_folder)
        f = open(out_path + "/" + name_in + "_" + name_out + ".ply", 'w')

        f.write("ply" + '\n')
        f.write("format ascii 1.0" + '\n')
        f.write("comment VCGLIB generated" + '\n')
        f.write("element vertex " + str(coords.shape[0]) + '\n')
        f.write("property float x" + '\n')
        f.write("property float y" + '\n')
        f.write("property float z" + '\n')
        f.write("element face 0" + '\n')
        f.write("property list uchar int vertex_indices" + '\n')
        f.write("end_header" + '\n')

        for row in range(coords.shape[0]):
            f.write(' '.join(map(str, coords[row, ...])) + '\n')

        f.close()


if __name__ == "__main__":
    main()
