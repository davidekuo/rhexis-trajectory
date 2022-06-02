from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from google.colab.patches import cv2_imshow
import cv2
import random
import os
import numpy as np
import warnings

"""
from eval_model import RhexisVisualizer
RhexisVis = RhexisVisualizer(bmc, cfg, object_detection_threshold = 0.98)

image_set = RhexisVis.randomly_sample_images_from_set("test",2)
RhexisVis.detectron2_visualizer(sampled_list = image_set)

RhexisVis.compare_labels(sampled_list = image_set)
"""


def convert_bbox(bbox):
  x0 = bbox[0]
  y0 = bbox[1]
  width = bbox[2] - bbox[0]
  height = bbox[3] - bbox[1]
  return [x0, y0, width, height]

def compare_bboxes(im, gt_bbox, pr_bbox):
  im = draw_bbox(im, gt_bbox, True)
  im = draw_bbox(im, pr_bbox, False)
  return im


def draw_bbox(im, bbox, ground = False):
  # Start coordinate, here (5, 5)
  # represents the top left corner of rectangle
  start_point = (round(bbox[0]), round(bbox[1]))
    
  # Ending coordinate, here (220, 220)
  # represents the bottom right corner of rectangle
  end_point = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
  
  # Blue color in BGR
  color = (0, 0, 0)
  if ground:
    color = (0, 255, 0)
  else:
    color = (255, 0, 0)
    
  # Line thickness of 2 px
  thickness = 1
    
  # Using cv2.rectangle() method
  # Draw a rectangle with blue line borders of thickness of 2 px
  im2 = cv2.rectangle(im, start_point, end_point, color, thickness)
  im = cv2.addWeighted(im, 0.95, im2, 0.05, 1.0)
  return im


def compare_keypoints(im, gt_keypoints, pr_keypoints):
  im = draw_keypoints(im, gt_keypoints, True)
  im = draw_keypoints(im, pr_keypoints, False)
  return im

def draw_keypoints(im, keypoints, ground = False):
  assert len(keypoints) % 3 == 0

  thickness = max(round(im.shape[0] / 300),1)

  for i in range(int(len(keypoints)/3)):
    xpos = round(keypoints[i*3])
    ypos = round(keypoints[i*3 + 1])
    color = (0,0,0)
    if ground:
      color = (0,255,0)
      print(f"Ground Truth Keypoint {i}: X {xpos}, Y {ypos}")
    else:
      color = (255,0,0)
      print(f"Predicted Keypoint {i}: X {xpos}, Y {ypos}")
    im = cv2.circle(im, (xpos,ypos), 10, color, thickness)
    im = cv2.circle(im, (xpos,ypos), 2, color, thickness)
  return im

class RhexisVisualizer:
  def __init__(self, model_weights_path, cfg, object_detection_threshold = 0.98):
    cfg.MODEL_WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = object_detection_threshold
    self.predictor = DefaultPredictor(cfg)

  def randomly_sample_images_from_set(self, set_to_sample = "test", num_images = 5):
    dataset_dicts = DatasetCatalog.get(set_to_sample)
    return random.sample(dataset_dicts, num_images)

  def detectron2_visualizer(self, set_to_sample = "test", num_images = 5, sampled_list = None):

    list_of_file_dicts = sampled_list
    if sampled_list is None:
      print(f"Randomly sampling {num_images} images from {set_to_sample} set")
      list_of_file_dicts = self.randomly_sample_images_from_set(set_to_sample, num_images)
    else:
      print(f"Showing {len(sampled_list)} files from past in list")

    for d in list_of_file_dicts:    
      im = cv2.imread(d["file_name"])
      outputs = self.predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        
      v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("train"), 
                    scale=0.5
      )
      out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      cv2_imshow(out.get_image()[:, :, ::-1])



  def compare_labels(self, set_to_sample = "test", num_images = 5, sampled_list = None):

    list_of_file_dicts = sampled_list
    if sampled_list is None:
      print(f"Randomly sampling {num_images} images from {set_to_sample} set")
      list_of_file_dicts = self.randomly_sample_images_from_set(set_to_sample, num_images)
    else:
      print(f"Showing {len(sampled_list)} files from past in list")

    for d in list_of_file_dicts:
      im = cv2.imread(d["file_name"])
      print("=========")
      print(f"File: {str.split(d['file_name'],os.sep)[-1]}")
      
      # Get the ground truth label data
      gt_bbox = d["annotations"][0]["bbox"]

      # Get the ground truth keypoints label data
      gt_keypoint = d["annotations"][0]['keypoints']

      # Calculate predictions
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = self.predictor(im)

      pr_bbox = None
      try:
        pr_bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy()[0]
        pr_bbox = convert_bbox(pr_bbox)
      except:
        print("No bbox")
        pass

      pr_keypoints = None
      try:
        pr_keypoints = outputs['instances'].pred_keypoints.cpu().numpy()[0]
        pr_keypoints = pr_keypoints.flatten()
      except:
        print("No keypoints")
        pass

      print("Ground Truth is green")
      print("Predictions are blue")
      if gt_bbox is not None and pr_bbox is not None:
        im = compare_bboxes(im, gt_bbox, pr_bbox)

      if gt_keypoint is not None and pr_keypoints is not None:
        im = compare_keypoints(im, gt_keypoint, pr_keypoints)

      cv2_imshow(im)