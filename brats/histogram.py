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
from os import listdir
import matplotlib

'''
script to generate histogram of spine size from ground truth or prediction

inputs:
 - ground truth or prediction of a case
outputs:
 - histogram of spine size

execution example:
 - python3 histogram.py --path_run "results/spines_dendrite/128x128x32_wloss_300ep" --name "prediction"

'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--name', default="prediction", help='working file name')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get path of the predictions
    name = parsed_args.name

    file = name + ".nii.gz"

    size = np.array([], dtype=float)  # init

    path_pred = os.path.join(path_run, "prediction")
    dir = listdir(path_pred)

    for case_folder in dir:

        print("Case: " + case_folder)

        # load file
        file_path = os.path.join(path_pred, case_folder, file)
        image = nib.load(file_path)
        data = image.get_data()

        den = np.where(data == [150])
        data[den[0], den[1], den[2]] = 0

        # get labels
        labels, num_labels = label(data)

        # get spines
        props = regionprops(labels)

        # read and save size
        for spine in range(num_labels):
            size_sp = props[spine].area
            size = np.append(size, size_sp)

    # make histogram of all case spine sizes and save it
    plt.figure()
    plt.hist(size, 300, [0, 10000])  # hist(data,bins,range)
    plt.title("Histogram")
    plt.savefig(path_run + '/hist_' + name +'.png')


if __name__ == "__main__":
    main()
