import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import sys


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='', help='path to the run folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run  # get data path

    path_file = os.path.join(path_run, "training.log")

    # visualice training details
    training_df = pd.read_csv(path_file).set_index('epoch')
    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig(path_run + '/loss_graph.png')

if __name__ == "__main__":
    main()
