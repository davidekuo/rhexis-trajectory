"""
Functions related to displaying keypoint and bounding box annotations.
"""
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
import rhexis_utils.edit_dataset_utils as edu
from google.colab.patches import cv2_imshow

def display_bboxes_comparisons(im, gt_bbox, pr_bbox):
    """
    Compares the ground truth and predicted bounding boxes and draws them on
    the image.

    Parameters:
        im(np.ndarray): The image to draw the bounding boxes on.
        gt_bbox(list): A list of four floats representing the ground truth bbox
            in the COCO format for bboxes.
            [<left top x>, <left top y>, <width>, <height>]
        
        pr_bbox(list): A list of four floats representing the predicted bbox
            in the COCO format for bboxes.
            [<left top x>, <left top y>, <width>, <height>]
        
    Returns:
        im(np.ndarray): The image with the bounding boxes drawn on it.
    """
    im = draw_bbox_on_image(im, gt_bbox, True)
    im = draw_bbox_on_image(im, pr_bbox, False)
    return im


def draw_bbox_on_image(im, bbox, ground=False):
    """
    Draws a bounding box on the image.

    Parameters:
        im(np.ndarray): The image to draw the bounding box on.

        bbox(list): A list of four floats representing the bounding box in the
            COCO format for bboxes.
            [<left top x>, <left top y>, <width>, <height>]
        
        ground(bool): Whether the bounding box is the ground truth or not.
            If ground is True, the bounding box will be green. Otherwise, it
            will be red.
            Default: False

    Returns:
        im(np.ndarray): The image with the bounding box drawn on it.
    """
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

def display_keypoint_comparisons(im, gt_keypoints, pr_keypoints):
    """
    Compares the ground truth and predicted keypoints and draws them on the
    image.
    
    Parameters:
        im(np.ndarray): The image to draw the keypoints on.
        
        gt_keypoints(list): A list of floats representing the ground truth
            keypoints in the COCO format for keypoints.
            [<x1>, <y1>, X, <x2>, <y2>, 2,...]
            
        pr_keypoints(list): A list of floats representing the predicted
            keypoints in the COCO format for keypoints.
            [<x1>, <y1>, 2, <x2>, <y2>, 2, ...]

        Note: In both the ground truth and predicted keypoints, the first two
            values are the x and y coordinates of the keypoint. The third
            value is the keypoint type. For our purposes, the third value
            should always be 2. TODO: Link to Detectron documentation
            
    Returns:
        im(np.ndarray): The image with the keypoints drawn on it.
    """
    im = draw_keypoints_on_image(im, gt_keypoints, True)
    im = draw_keypoints_on_image(im, pr_keypoints, False)
    return im


def draw_keypoints_on_image(im, keypoints, ground=False):
    """
    Draws keypoints on the image.

    Parameters:
        im(np.ndarray): The image to draw the keypoints on.

        keypoints(list): A list of floats representing the keypoints in the
            COCO format for keypoints.
            [<x1>, <y1>, 2, <x2>, <y2>, 2, ...]

        ground(bool): Whether the keypoints are the ground truth or not.
            If ground is True, the keypoints will be green. Otherwise, they
            will be red.
            Default: False
    
    Returns:
        im(np.ndarray): The image with the keypoints drawn on it.
    """

    assert len(keypoints) % 3 == 0

    thickness = max(round(im.shape[0] / 300), 1)

    for i in range(int(len(keypoints) / 3)):
        xpos = round(keypoints[i * 3])
        ypos = round(keypoints[i * 3 + 1])
        color = (0, 0, 0)
        if ground:
            color = (0, 255, 0)
            print(f"Ground Truth Keypoint {i}: X {xpos}, Y {ypos}")
        else:
            color = (255, 0, 0)
            print(f"Predicted Keypoint {i}: X {xpos}, Y {ypos}")
        im = cv2.circle(im, (xpos, ypos), 10, color, thickness)
        im = cv2.circle(im, (xpos, ypos), 2, color, thickness)
    return im


def display_annotations(test_subdir, test_index, DATA_LOC):
    """
    Displays the annotations for the image at the given index of the dataset.

    Parameters:
        test_subdir(str): The subdirectory of the test dataset.
        
        test_index(int): The index of the image in the test dataset.

        DATA_LOC(str): The path to the dataset.
    """

    # Read in json dict
    json_dict = edu.read_in_json(edu.find_json_file(DATA_LOC, test_subdir))

    # Get image dict
    image_dict = json_dict["images"][test_index]

    # Read in the image
    image_path = os.path.join(DATA_LOC, test_subdir, "images", image_dict["file_name"])
    print(image_path)

    # print id
    print(f"Image ID: {image_dict['id']}")

    print("=JSON_DATA=")
    print("images")
    print(image_dict)

    im = cv2.imread(image_path)
    # Find search annotation
    search_annotate_dict = edu.create_search_annotate_dict(json_dict["annotations"])

    # Use search dict to find annotations for this image
    annotate_dict = None
    try:
        annotate_dict = search_annotate_dict[image_dict["id"]]
    except KeyError:
        pass
        # print(f"Image {im_fn} does not have annotations")

    # only correct annotations if they exist for this image
    if annotate_dict is not None:
        print("annotations")
        print(annotate_dict)
        # First, correct bbox and area annotations (if they exist)
        bbox = None
        try:
            bbox = annotate_dict["bbox"]
        except KeyError:
            pass

        if bbox is not None:
            print(f"bbox: {bbox}")
            im = draw_bbox_on_image(im, bbox)

        # Now draw keypoints (if they exist)
        keypoints = None
        try:
            keypoints = annotate_dict["keypoints"]
        except KeyError:
            pass

        if keypoints is not None:
            print(f"keypoints: {keypoints}")
            im = draw_keypoints_on_image(im, keypoints)

    print(f"Image size:{im.shape}")
    cv2_imshow(im)

def display_annotations_data_edit(
    test_subdir_nickname, test_index, DATA_LOC, use_original_json=False
):
    """
    TODO: MERGE THIS FUNCTIONALITY WITH display_resized_annotation
    """
    test_subdir = test_subdir_nickname + "_set"

    # Read in json dict
    if use_original_json:
        json_dict = edu.read_in_json(
            os.path.join(
                DATA_LOC,
                "original_jsons",
                test_subdir_nickname + "_coco_annotations.json",
            )
        )
    else:
        json_dict = edu.read_in_json(edu.find_json_file(DATA_LOC, test_subdir))

    # Get image dict
    image_dict = json_dict["images"][test_index]

    # Read in the image
    image_path = os.path.join(DATA_LOC, test_subdir, "images", image_dict["file_name"])
    print(image_path)

    # print id
    print(f"Image ID: {image_dict['id']}")

    print("=JSON_DATA=")
    print("images")
    print(image_dict)

    im = cv2.imread(image_path)
    # Find search annotation
    search_annotate_dict = edu.create_search_annotate_dict(json_dict["annotations"])

    # Use search dict to find annotations for this image
    annotate_dict = None
    try:
        annotate_dict = search_annotate_dict[image_dict["id"]]
    except KeyError:
        pass
        # print(f"Image {im_fn} does not have annotations")

    # only correct annotations if they exist for this image
    if annotate_dict is not None:
        print("annotations")
        print(annotate_dict)
        # First, correct bbox and area annotations (if they exist)
        bbox = None
        try:
            bbox = annotate_dict["bbox"]
        except KeyError:
            pass

        if bbox is not None:
            print(f"bbox: {bbox}")
            im = draw_bbox_on_image(im, bbox)

        # Now draw keypoints (if they exist)
        keypoints = None
        try:
            keypoints = annotate_dict["keypoints"]
        except KeyError:
            pass

        if keypoints is not None:
            print(f"keypoints: {keypoints}")
            im = draw_keypoints_on_image(im, keypoints)

    print(f"Image size:{im.shape}")
    cv2_imshow(im)

def display_resized_annotation(test_subdir, test_index, DATA_LOC, NEW_DATA_LOC):
    """
    Display resized annotations for the image at the given index of the dataset.

    Parameters:
        test_subdir(str): The subdirectory of the test dataset.

        test_index(int): The index of the image in the test dataset we wish to
            display.

        DATA_LOC(str): The path to the dataset.

        NEW_DATA_LOC(str): The path to the resized dataset.
    """

    # Call display for original image
    print("ORIGINAL IMAGE")
    display_annotations(test_subdir, test_index, DATA_LOC)

    # Call display for resized image
    print("RESIZED IMAGE")
    display_annotations(test_subdir, test_index, NEW_DATA_LOC)
