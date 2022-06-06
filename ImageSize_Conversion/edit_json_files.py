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
from test_utils import *
import resize_utils as ru


def reset_jsons(DATA_LOC):
    # Get the other json files
    test_json = os.path.join(DATA_LOC, "original_jsons", "test_coco_annotations.json")
    train_json = os.path.join(DATA_LOC, "original_jsons", "train_coco_annotations.json")
    val_json = os.path.join(DATA_LOC, "original_jsons", "val_coco_annotations.json")

    # Solve path to edited jsons
    test_ej = find_json_file(DATA_LOC, "test_set")
    train_ej = find_json_file(DATA_LOC, "train_set")
    val_ej = find_json_file(DATA_LOC, "val_set")

    original_jsons = [test_json, train_json, val_json]
    edited_jsons = [test_ej, train_ej, val_ej]

    # Overwrite the edited jsons
    for oj, ej in zip(original_jsons, edited_jsons):
        with open(oj) as ojf:
            json_dict = json.load(ojf)
            json_string = json.dumps(json_dict)
            with open(ej, "w") as outfile:
                outfile.write(json_string)


def grow_bbox(imshape, bbox, num_pixels):
    assert num_pixels >= 0
    width, height = imshape

    # bbox format: [top left x position, top left y position, width, height]
    new_bbox = bbox

    # Subtract from left point x
    new_bbox[0] -= num_pixels
    if new_bbox[0] < 0:
        new_bbox[0] = 0

    # Subtract from left point y
    new_bbox[1] -= num_pixels
    if new_bbox[1] < 0:
        new_bbox[1] = 0

    # Add to width
    new_bbox[2] += num_pixels
    if new_bbox[0] + new_bbox[2] > width:
        new_bbox[2] = width - new_bbox[0]

    # Add to height
    new_bbox[3] += num_pixels
    if new_bbox[1] + new_bbox[3] > width:
        new_bbox[3] = width - new_bbox[1]

    # replace area
    new_area = new_bbox[2] * new_bbox[3]

    return new_bbox, new_area


def edit_jsons_bboxs_all(DATA_LOC, subdir_names, num_pixels=2.0):
    json_dict_list = []
    for subdir_name in subdir_names:
        json_dict_list.append(edit_json_bboxs(DATA_LOC, subdir_name, num_pixels))

    return json_dict_list


def edit_json_bboxs(DATA_LOC, subdir_name, num_pixels=2.0):

    # Find json file
    json_filename = find_json_file(DATA_LOC, subdir_name)

    # Get json data
    json_dict = read_in_json(json_filename)

    # Get image dict list
    image_dict_list = json_dict["images"]

    # get annotation dict list
    annotate_dict_list = json_dict["annotations"]
    search_annotate_dict = create_search_annotate_dict(annotate_dict_list)

    for image_dict in image_dict_list:
        im_id = image_dict["id"]
        # Use search dict to find annotations for this image
        annotate_dict = None
        try:
            annotate_dict = search_annotate_dict[im_id]
        except KeyError:
            pass
            # print(f"Image {im_fn} does not have annotations")

        if annotate_dict is not None:
            # First, correct bbox and area annotations (if they exist)
            bbox = None
            try:
                bbox = annotate_dict["bbox"]
            except KeyError:
                pass

            # if we have a bbox, set the new bbox and area
            if bbox is not None:
                imshape = (image_dict["width"], image_dict["height"])
                bbox_new, area_new = grow_bbox(imshape, bbox, num_pixels)
                # set in dict
                annotate_dict["bbox"] = bbox_new
                annotate_dict["area"] = area_new

    # Write changes to the json file
    # Create new json file
    json_string = json.dumps(json_dict)
    with open(json_filename, "w") as outfile:
        outfile.write(json_string)

    return json_dict


def create_search_annotate_dict(annotate_dict_list):
    search_annotate_dict = {}
    for annotate_dict in annotate_dict_list:
        an_id = annotate_dict["image_id"]
        search_annotate_dict.setdefault(an_id, annotate_dict)
    return search_annotate_dict


def find_json_file(DATA_LOC: str, subdir_name: str):
    # Find the json file
    json_files = glob.glob(os.path.join(DATA_LOC, subdir_name, "*.json"))

    # Assert that only one json is found
    assert len(json_files) > 0, f"No json file found in {subdir_name}"
    assert len(json_files) == 1, f"More than one json file found in {subdir_name}"

    # Return json file
    return json_files[0]


def read_in_json(json_filename: str):
    with open(json_filename) as jf:
        data = json.load(jf)
        return data


def display_annotations_data_edit(
    test_subdir_nickname, test_index, DATA_LOC, use_original_json=False
):
    test_subdir = test_subdir_nickname + "_set"

    # Read in json dict
    if use_original_json:
        json_dict = ru.read_in_json(
            os.path.join(
                DATA_LOC,
                "original_jsons",
                test_subdir_nickname + "_coco_annotations.json",
            )
        )
    else:
        json_dict = ru.read_in_json(ru.find_json_file(DATA_LOC, test_subdir))

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
    search_annotate_dict = ru.create_search_annotate_dict(json_dict["annotations"])

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
            im = draw_bbox(im, bbox)

        # Now draw keypoints (if they exist)
        keypoints = None
        try:
            keypoints = annotate_dict["keypoints"]
        except KeyError:
            pass

        if keypoints is not None:
            print(f"keypoints: {keypoints}")
            im = draw_keypoints(im, keypoints)

    print(f"Image size:{im.shape}")
    cv2_imshow(im)
