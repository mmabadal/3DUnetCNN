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
    
   name = parsed_args.name

   file = name + ".nii.gz"
    
   sizex = np.array([], dtype=int)
   sizey = np.array([], dtype=int)
   sizez = np.array([], dtype=int)
   size = np.array([], dtype=int)

   path_pred = parsed_args.path_run  # get path of the predictions

   #path_pred = os.path.join(path_run, "prediction")
   dir = listdir(path_pred)

   for case_folder in dir:

      print("Case: " + case_folder)

      # load file
      file_path = os.path.join(path_pred, case_folder, file)
      image = nib.load(file_path)
      data = image.get_data()
    
      # delete dendrite
      den = np.where(data == [150])
      data[den[0], den[1], den[2]] = 0

      # get labels
      labels, num_labels = label(data)

      # get gt and predictions spines
      props = regionprops(labels)

      # get the x,y,z coordinates from the bounding box related to each spine
      for spine in range(num_labels):
         spine_bb = props[spine].bbox
         spine_x=spine_bb[4]-spine_bb[1]
         sizex = np.append(sizex, spine_x)
         spine_y=spine_bb[3]-spine_bb[0]
         sizey = np.append(sizey, spine_y)
         spine_z=spine_bb[5]-spine_bb[2]
         sizez = np.append(sizez, spine_z)
         size_sp = props[spine].area
         size = np.append(size, size_sp)
   
   # get the histogram
   '''
   plt.figure(figsize=(16, 24))

   plt.subplot(4, 1, 1)
   plt.hist(sizex, 100, [0, 100])  # hist(data,bins,range)
   plt.minorticks_on()
   plt.xlabel('x size of the spine')
   plt.ylabel('Number of spines')

   plt.subplot(4, 1, 2)
   plt.hist(sizey, 100, [0, 100])  # hist(data,bins,range)
   plt.minorticks_on()
   plt.xlabel('y size of the spine')
   plt.ylabel('Number of spines')

   plt.subplot(4, 1, 3)
   plt.hist(sizez, 100, [0, 100])  # hist(data,bins,range)
   plt.minorticks_on()
   plt.xlabel('z size of the spine')
   plt.ylabel('Number of spines')

   plt.subplot(4, 1, 4)
   '''
   plt.hist(size, 300, [0, 500])  # hist(data,bins,range)
   plt.minorticks_on()
   plt.xlabel('spine area')
   plt.ylabel('Number of spines')

   plt.savefig(path_pred + '/Histogram_' + name + '.pdf')



if __name__ == "__main__":
    main()
