"""
Functions to access segmentation from saved PNG images.
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
  """
  Returns an numpy array of shape HxWxC in BGR (using cv2.imread)

  Parameters:
    filename_substring: A substring of the file you would like to get the image
      data of.
    DATA_LOC: the path to the data in your Google Drive

  Returns: A numpy array of the requested image
  """
  return filename_to_numpy(get_full_image_filename(filename_substring, DATA_LOC))


def filename_to_numpy(full_file_path: str):
  """ 
  Returns a numpy version of the requested image at full_file_path
    in BGR numpy (HxWxC)

  Parameters:
    full_file_path: A string containing a full file path to the image you would
      like to read in

  Returns: A numpy array of the requested image.
  """
  # Collect the image
  img = cv2.imread(full_file_path)

  # Cut off unneeded channels and return
  return img

def label_to_numpy(full_file_path: str):
  """ 
  Returns a numpy version of the requested segmentation label at 
    full_file_path in as a 2D numpy array (HxW)

  Parameters:
    full_file_path: A string containing a full file path to the image you would
    like to read in.

  Returns: A numpy array of the requested label segmentation.
  """
  return filename_to_numpy(full_file_path)[:,:,0]

def get_full_image_filename(filename_substring: str, DATA_LOC: str):
  """
  Gets the full image filename from any substring. Will fail if multiple image
  filenames contain the specified substring or if no image filenames contain the
  specified substring.

  Parameters:
    filename_substring: A substring of the file we would like returned

    DATA_LOC: The path to the directory containing the datasets

  Returns:
    Full filename of the requested file
  """
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
  Returns a numpy array of the label that corresponds to the requested image.

  Parameters:
    filename_substring: A substring of the image file we would like to get a
      label for
    
    DATA_LOC: A path to the directory containing the dataset folders

  Returns:
    2D numpy array of the label (HxW)
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

  Parameters:
    task: Task is an integer (between 1 and 3) that contains the task we are
      running. (Different tasks have different segmented classes) Default is 2.

  Returns: A dict containing all of the labels. Keys are label integers
    corresponding to the values in the label pngs and values are string names
    that specify what the labels correspond to ("Pupil" etc)
  """
  return globals()[f'classes_exp{task}']
