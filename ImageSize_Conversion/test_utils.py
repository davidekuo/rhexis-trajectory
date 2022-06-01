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
import resize_utils as ru
from google.colab.patches import cv2_imshow

def draw_keypoints(im, keypoints):
  assert len(keypoints) % 3 == 0

  for i in range(int(len(keypoints)/3)):
    xpos = round(keypoints[i*3])
    ypos = round(keypoints[i*3 + 1])
    im = cv2.circle(im, (xpos,ypos), 2, (255,0,0), 5)

  return im


def draw_bbox(im, bbox):
  # Start coordinate, here (5, 5)
  # represents the top left corner of rectangle
  start_point = (round(bbox[0]), round(bbox[1]))
    
  # Ending coordinate, here (220, 220)
  # represents the bottom right corner of rectangle
  end_point = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
  
  # Blue color in BGR
  color = (0, 0, 255)
    
  # Line thickness of 2 px
  thickness = 2
    
  # Using cv2.rectangle() method
  # Draw a rectangle with blue line borders of thickness of 2 px
  im = cv2.rectangle(im, start_point, end_point, color, thickness)
  return im


def display_annotations(test_subdir, test_index, DATA_LOC):

  # Read in json dict
  json_dict = ru.read_in_json(ru.find_json_file(DATA_LOC, test_subdir))


  # Get image dict
  image_dict = json_dict['images'][test_index]


  # Read in the image
  image_path = os.path.join(DATA_LOC, test_subdir, "images", image_dict['file_name'])
  print(image_path)

  # print id
  print(f"Image ID: {image_dict['id']}")

  print("=JSON_DATA=")
  print("images")
  print(image_dict)

  im = cv2.imread(image_path)
  # Find search annotation
  search_annotate_dict = ru.create_search_annotate_dict(json_dict['annotations'])

  # Use search dict to find annotations for this image
  annotate_dict = None
  try:
    annotate_dict = search_annotate_dict[image_dict['id']]
  except KeyError:
    pass
    #print(f"Image {im_fn} does not have annotations")
    

  # only correct annotations if they exist for this image
  if annotate_dict is not None:
    print("annotations")
    print(annotate_dict)
    # First, correct bbox and area annotations (if they exist)
    bbox = None
    try:
      bbox = annotate_dict['bbox']
    except KeyError:
      pass

    if bbox is not None:
      print(f"bbox: {bbox}")
      im = draw_bbox(im, bbox)

    # Now draw keypoints (if they exist)
    keypoints = None
    try:
      keypoints = annotate_dict['keypoints']
    except KeyError:
      pass

    if keypoints is not None:
      print(f"keypoints: {keypoints}")
      im = draw_keypoints(im, keypoints)

  print(f"Image size:{im.shape}")
  cv2_imshow(im)



def display_annotations_both(test_subdir, test_index, DATA_LOC, NEW_DATA_LOC):

  # Call display for original image
  print("ORIGINAL IMAGE")
  display_annotations(test_subdir, test_index, DATA_LOC)

  # Call display for resized image
  print("RESIZED IMAGE")
  display_annotations(test_subdir, test_index, NEW_DATA_LOC)