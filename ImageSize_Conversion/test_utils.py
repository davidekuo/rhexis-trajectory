# Semantic Segmentation Utils
import os
import json
import pathlib
import torch
import numpy as np
import glob
import cv2
import time
import math
import json


def create_img_data_dict(img_data, img_data_dict):
  for i, filename in enumerate(img_data[0]):
    sizes = img_data[1][i][:]

    # Cure filename to only include last part
    name = str.split(filename,os.sep)[-1]
    img_data_dict.setdefault(name, sizes)

def create_search_annotate_dict(annotate_dict_list):
  search_annotate_dict = {}
  for annotate_dict in annotate_dict_list:
    an_id = annotate_dict['id']
    search_annotate_dict.setdefault(an_id, annotate_dict)
  return search_annotate_dict



def caculate_scales(ori_size, new_size):
  # Determine how much we should scale
  height_scale = new_size[0] / ori_size[0]
  width_scale = new_size[1] / new_size[0]

  return height_scale, width_scale

def correct_keypoint(keypoint_list_original, ori_size, new_size):
  # Determine the scales
  height_scale, width_scale = caculate_scales(ori_size, new_size)

  # For each keypoint, there are 3 numbers
  assert len(keypoint_list_original) % 3 == 0

  keypoint_list_new = keypoint_list_original
  for i in range(int(len(keypoint_list_original)/3)):
    xpos = keypoint_list_original[i*3]
    ypos = keypoint_list_original[i*3 + 1]
    # dont't change the third element (2)

    # Scale and update
    keypoint_list_new[i*3] = round(xpos * width_scale, 2)
    keypoint_list_new[i*3 + 1] = round(ypos * height_scale, 2)
    # dont't change the third element (2)

  return keypoint_list_new

def correct_bbox(bbox_original, ori_size, new_size):
  # bbox format: [top left x position, top left y position, width, height]

  # Determine scales
  height_scale, width_scale = caculate_scales(ori_size, new_size)

  # Scale top left position
  top_left_x = round(width_scale * bbox_original[0],2)
  top_left_y = round(height_scale * bbox_original[1],2)

  # Scale width
  width = round(width_scale * bbox_original[2],2)
  height = round(height_scale * bbox_original[2],2)

  # Set new bbox
  new_bbox = [top_left_x, top_left_y, width, height]

  # Determine new area
  new_area = width * height

  return new_bbox, new_area




def correct_all_json_files(STANDARD_SIZE, DATA_LOC : str, NEW_DATA_LOC : str, img_data, subdir_names = []):
  # store returns in list
  json_dict_list = []
  for name in subdir_names:
    json_dict_list.append(correct_json_file(STANDARD_SIZE, DATA_LOC, NEW_DATA_LOC, img_data, name))
  
  return json_dict_list

def correct_json_file(STANDARD_SIZE, DATA_LOC : str, NEW_DATA_LOC : str, img_data, subdir_name : str):
  # Find the json file
  json_files = glob.glob(os.path.join(DATA_LOC, subdir_name, "*.json"))

  # Assert that only one json is found
  assert len(json_files) > 0, f"No json file found in {subdir_name}"
  assert len(json_files) == 1, f"More than one json file found in {subdir_name}"

  # Start by reading in the json file
  json_filename = json_files[0]

  # Get the json_dict
  json_dict = read_in_json(json_filename)

  # Convert the img_data structure to a dict for easy access to previous sizes
  img_data_dict = {}
  create_img_data_dict(img_data, img_data_dict)

  # EDIT JSON DICT
  image_dict_list = json_dict['images']
  annotate_dict_list = json_dict['annotations']

  # Copy of annotate dict list to optimize search
  search_annotate_dict = create_search_annotate_dict(annotate_dict_list.copy())

  # For each image in image_dict_list
  for image_dict in image_dict_list:

    # Pull vital info from image_dict
    im_id = image_dict['id']
    im_fn = image_dict['file_name']

    # Pull vital info from img_data (1 0 is correct order here)
    ori_width = img_data_dict[im_fn][1]
    ori_height = img_data_dict[im_fn][0]
    
    # Set new width and height (0 1 is correct order here)
    image_dict['width'] = STANDARD_SIZE[0]
    image_dict['height'] = STANDARD_SIZE[1]

    # Use search dict to find annotations for this image
    annotate_dict = None
    try:
      annotate_dict = search_annotate_dict[im_id]
    except KeyError:
      pass
      #print(f"Image {im_fn} does not have annotations")

    # only correct annotations if they exist for this image
    if annotate_dict is not None:
      # First, correct bbox and area annotations (if they exist)
      bbox_original = None
      try:
        bbox_original = annotate_dict['bbox']
      except KeyError:
        pass
      
      # if we have a bbox, set the new bbox and area
      if bbox_original is not None:
        # Set new bbox and area
        ori_size = (ori_height, ori_width)
        bbox_new, area_new = correct_bbox(bbox_original, ori_size, STANDARD_SIZE)

        # set in dict
        annotate_dict['bbox'] = bbox_new
        annotate_dict['area'] = area_new
      
      # Now attempt to do the same thing for the keypoints
      keypoint_list = None
      try:
        keypoint_list = annotate_dict['keypoints']
      except KeyError:
        pass
      
      
      if keypoint_list is not None:
        new_keypoint_list = correct_keypoint(keypoint_list, ori_size, STANDARD_SIZE)

        annotate_dict['keypoints'] = new_keypoint_list
  
  # Get json_dict file name
  json_file_shortname = str.split(json_filename, os.sep)[-1]

  # Emphasize new json filename
  new_json_filename = os.path.join(NEW_DATA_LOC, subdir_name, json_file_shortname)

  # Create new json file
  json_string = json.dumps(json_dict)
  with open(new_json_filename, 'w') as outfile:
    outfile.write(json_string)

  return json_dict


def read_in_json(json_filename:str):
  with open(json_filename) as jf:
    data = json.load(jf)
    return data


def make_resized_image_directories(NEW_DATA_LOC, subdir_names = []):

  if not subdir_names:
    subdir_names = ['train_set', 'val_set', 'test_set']
  
  # Create these folders
  for name in subdir_names:
    folder_to_create = os.path.join(NEW_DATA_LOC, name, "images")
    
    try:
      os.makedirs(folder_to_create)
    except OSError as error:
      pass
      print(f"{name} label folder already exists")

def create_resized_images(images, img_data, NEW_DATA_LOC, subdir_names = []):
  """
  Creates resized images of all of our images
  """
  # Keep track of how many images saved successfully and which ones did not
  success_count = 0

  # Keeps track of which images
  failures = []

  # Unpack img_data
  file_list, size_list = img_data

  # Make the label directories, if they do not exist
  make_resized_image_directories(NEW_DATA_LOC, subdir_names)

  # For each image
  for i in range(images.shape[0]):
    # Get the current file name
    f = file_list[i]

    # Determine the name of this image
    file_name = str.split(str.split(f, os.sep)[-1],".")[0]
    
    # Determine which set the image is in
    image_set = str.split(f, os.sep)[-3]

    # Create final image path
    image_path = os.path.join(NEW_DATA_LOC, image_set, "images", file_name + ".jpg")

    # Write image to path location
    if(cv2.imwrite(image_path, images[i])):
      success_count += 1
    else:
      failures.append(f)

  if len(failures) > 0:
    print("Images that failed to generate resized images:")
    for fail in failures:
      print(fail)
  
  print("")
  print(f"Successfully saved {success_count} resized images")
  print("")
  return success_count


def read_in_images(DATA_LOC: str, STANDARD_SIZE,subdir_names = []):
  """
  Reads in images from the dataset location and returns a numpy array containing
  the images of shape (NxHxWxC), where N is the number of images, C is the
  number of channels, and H + W are sizes of spatial dimensions.
  """

  if not subdir_names:
    subdir_names = ['train_set', 'val_set', 'test_set']

  # Glob the data together
  X = []

  # Collect all *.jpg files in sub-directories
  file_list = []
  for name in subdir_names:
    files_at_name_folder = glob.glob(os.path.join(DATA_LOC, name, "images", "*"))
    file_list = file_list + files_at_name_folder

  # save the sizes of each file so we can resize at the end
  size_list = []
  print("Loading in and resizing images:")
  print(f"{len(file_list)} image files detected")
  for f in file_list:
    image= cv2.imread(f)
    size_list.append((image.shape[0],image.shape[1]))
    X.append(np.array(cv2.resize(image, (STANDARD_SIZE[0], STANDARD_SIZE[1]))))


  # Stack images into 4D NxHxWxC
  stack = np.array(X)

  print()
  print("Images loaded and resized successfully")
  print(f"Returned stack has size NxHxWxC: {stack.shape}")
  return stack, (file_list, size_list)