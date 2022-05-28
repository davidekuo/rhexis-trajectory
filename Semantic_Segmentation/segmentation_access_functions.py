"""
Functions to access segmentations
"""
import glob
import os
import cv2
from utils import *
from skimage import io
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
import numpy as np


def get_image_from_image_filename(filename_substring: str, DATA_LOC: str):
  return filename_to_numpy(get_full_image_filename(filename_substring, DATA_LOC))


def filename_to_numpy(full_file_path: str):
  """ 
  Returns a numpy version of the requested image at full_file_path
  """
  # Collect the image
  img = cv2.imread(full_file_path)

  # Cut off unneeded channels and return
  return img

def label_to_numpy(full_file_path: str):
  return filename_to_numpy(full_file_path)[:,:,0]

def get_full_image_filename(filename_substring: str, DATA_LOC: str):
  # First, determine the full file path of the requested image
  # Collect all *.jpg files
  train_files = glob.glob(os.path.join(DATA_LOC, "train_set","images","*.jpg"))
  val_files = glob.glob(os.path.join(DATA_LOC, "val_set","images","*.jpg"))
  test_files = glob.glob(os.path.join(DATA_LOC, "test_set","images","*.jpg"))

  # Concatentate the list of files
  file_list = train_files + val_files + test_files

  # find the requested file in the list of files
  files_found = [f for f in file_list if filename_substring in f]

  # if more than one file was found, throw an error
  if len(files_found) > 1:
    print("Error: Multiple files match substring")
    for f in files_found:
      print(f)
    assert False, f"More than one image filename contains the substring '{filename_substring}'"

  # If no file was found, throw an error
  assert len(files_found) == 1, f"No image filename that contains the substring '{filename_substring}' was found"

  return files_found[0]



def get_label_from_image_filename(filename_substring: str, DATA_LOC: str):
  """
  Returns a numpy array of the label that corresponds to the requested image
  """

  # See if we can find this image file
  file_found = get_full_image_filename(filename_substring, DATA_LOC)


  # Determine the png label filename and the set the label is in
  file_name_png = str.split(str.split(file_found,os.sep)[-1],'.')[0] + ".png"
  label_set = str.split(file_found,os.sep)[-3]

  # Get full label path from full image file path
  full_label_path = os.path.join(DATA_LOC, label_set, "labels",file_name_png)


  # Return a numpy version of the image
  return label_to_numpy(full_label_path)


def get_labels(task = 2):
  """
  Retruns a dict of the labels in a segmentation
  """
  return globals()[f'classes_exp{task}']
