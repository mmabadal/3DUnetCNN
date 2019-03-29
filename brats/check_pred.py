import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import argparse
from os import listdir
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
import matplotlib


prediction = nib.load('prediction.nii.gz')


pred = prediction.get_data()

u = np.unique(pred)

z=1
