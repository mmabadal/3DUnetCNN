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
 - python3 export_ply.py --path_run "results/spines_dendrite/64x64x64_da_medium_300_wdl_sigmoid/" --rescale 0 --name "prediction"

 - python3 export_ply.py --path_run "results/spines_dendrite/64x64x64_da_medium_300_wdl_sigmoid/" --rescale 0 --name "threshold"
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--rescale', default=0, help='1 to rescale.')
    parser.add_argument('--name', help='working file name')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get prediction folder
    rescale = parsed_args.rescale  # get rescale option
    name = parsed_args.name  # get type

    file = name + ".nii.gz"

    path_pred = os.path.join(path_run, "prediction")

    dir = listdir(path_pred)

    for case_folder in dir:

        # load nii file
        file_path = os.path.join(path_pred, case_folder, file)
        prediction = nib.load(file_path)
        prediction = prediction.get_data()

        pred = np.where(prediction != [0])  # get detected index

        # get detected coords

        x = pred[0]
        y = pred[1]
        z = pred[2]
        coords = np.zeros((x.shape[0], 3), dtype=int)  # auxiliary image

        # rescale if indicated
        if rescale == 1:
            x = 92.803 + 0.0751562 * x
            y = -51.543 + 0.0751562 * y
            z = 50.927 + 0.279911 * z
            coords = np.zeros((x.shape[0], 3))  # auxiliary image

        # append information
        for coord in range(x.shape[0]):
            coords[coord, 0] = int(x[coord])
            coords[coord, 1] = int(y[coord])
            coords[coord, 2] = int(z[coord])

        # create ply
        out_path = os.path.join(path_pred, case_folder)
        f = open(out_path + "/" + name + ".ply", 'w')

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
