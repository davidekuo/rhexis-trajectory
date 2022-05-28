"""
Functions to access segmentations
"""
import glob
import os
import cv2

def label_to_numpy(full_file_path: str):
  label_image = cv2.imread(full_file_path)
  return label_image


def get_label_from_image_filename(filename_substring: str, DATA_LOC: str):
  """
  Returns a numpy array of the label that corresponds to the requested image
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
    assert False, f"More than one image filename contains the substring '{filename_substring}', can not map to label"

  # If no file was found, throw an error
  assert len(files_found) == 1, f"No image filename that contains the substring '{filename_substring}' was found, can not map to label."

  # Determine the png label filename and the set the label is in
  file_name_png = str.split(str.split(files_found[0],os.sep)[-1],'.')[0] + ".png"
  label_set = str.split(files_found[0],os.sep)[-3]

  # Get full label path from full image file path
  full_label_path = os.path.join(DATA_LOC, label_set, "labels",file_name_png)


  # Return a numpy version of the image
  return label_to_numpy(full_label_path)
