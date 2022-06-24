"""
A module of utility functions datasets of the Rhexis Project.
"""

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

def reset_jsons(DATA_LOC):
    """
    Resets the JSON annotation files to the ones in the original dataset.

    Parameters:
        DATA_LOC:(str) The location of the main dataset directory.
    """
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


def grow_bbox(imshape : tuple, bbox :list, num_pixels : int):
    """
    Increases the size of the bounding box by moving the top left position
    of the bounding box up by num_pixels and to the left by num_pixels.

    Parameters:
        imshape: (tuple) The shape of the image.

        bbox: The original bounding box.
            [<top L x>, <top L y>, <width>, <height>]

        num_pixels: (int) The number of pixels to move the left corner of the
            bounding box by.

    Returns:
        new_bbox: The resized bounding box.
            [<top L x>, <top L y>, <width>, <height>]
        
        new_area: The new area of the bounding box.
    """
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


def grow_subdir_bboxs_all(DATA_LOC, subdir_names, num_pixels=2.0):
    """
    Grows the bounding boxes of all images in all subdirectories of the dataset
    by num_pixels.

    Parameters:
        DATA_LOC: (str) The location of the main dataset directory.

        subdir_names: (list) The names of the subdirectories to grow the
            bounding boxes of.

        num_pixels: (int) The number of pixels to move the left corner of the
            bounding box by.
        
    Returns:
        json_dict_list: (list) A list of dictionaries containing the updated
            JSON annotations of the bounding boxes.
    """
    json_dict_list = []
    for subdir_name in subdir_names:
        json_dict_list.append(grow_subdir_bboxs(DATA_LOC, subdir_name, num_pixels))

    return json_dict_list


def grow_subdir_bboxs(DATA_LOC, subdir_name, num_pixels=2.0):
    """
    Grow the bounding boxes of all images in a subdirectory of the dataset
    by num_pixels.

    Parameters:
        DATA_LOC: (str) The location of the main dataset directory.

        subdir_name: (str) The name of the subdirectory to grow the bounding
            boxes of.
    
        num_pixels: (int) The number of pixels to move the left corner of the
            bounding box by. (moves both up and to the left by num_pixels)

    Returns:
        json_dict: (dict) A dictionary containing the updated JSON annotations
            of the bounding boxes.
    """

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
    """
    Creates a dictionary that can be used to easily access the annotations
    of a particular image.

    Parameters:
        annotate_dict_list: (list) A list of dictionaries containing the
            annotations of the images.

    Returns:
        search_annotate_dict: (dict) A dictionary that can be used to easily
            access the annotations of a particular image. Keys are the image
            IDs and values are the annotations.
    """
    search_annotate_dict = {}
    for annotate_dict in annotate_dict_list:
        an_id = annotate_dict["image_id"]
        search_annotate_dict.setdefault(an_id, annotate_dict)
    return search_annotate_dict


def create_img_data_dict(img_data, img_data_dict):
    """
    Creates a dictionary of image data stored in img_data_dict.

    Parameters:
        img_data: list of dictionaries containing image data from JSON files
        img_data_dict: A single dictionary containing all image data with the 
            name of the image as the key.
    """
    for i, filename in enumerate(img_data[0]):
        sizes = img_data[1][i][:]

        # Cure filename to only include last part
        name = str.split(filename, os.sep)[-1]
        img_data_dict.setdefault(name, sizes)

def caculate_scales(ori_size, new_size):
    """
    Determines the relative scale of the horizontal and vertical edges of the
    original and scaled images.

    Parameters:
        ori_size: tuple of ints
            The size of the original image. (Height, Width)

        new_size: tuple of ints
            The size of the new image. (Height, Width)
        
    Returns:
        height_scale: float
            The relative scale of the height of the images (new / original)

        width_scale: float
            The relative scale of the width of the images (new / original)
    """
    # Determine how much we should scale
    height_scale = new_size[0] / ori_size[0]
    width_scale = new_size[1] / ori_size[1]

    return height_scale, width_scale


def rescale_keypoints(keypoint_list_original, ori_size, new_size):
    """
    Rescales the keypoints to the correct locations in the rescaled image.

    Parameters:
        keypoint_list_original: list of keypoint coordinates

        ori_size: tuple of ints
            The size of the original image. (Height, Width)

        new_size: tuple of ints
            The size of the new image. (Height, Width)

    Returns:
        keypoint_list_new: list of keypoint coordinates rescaled to their locations
            on the new rescaled image.
    """
    # Determine the scales
    height_scale, width_scale = caculate_scales(ori_size, new_size)

    # For each keypoint, there are 3 numbers
    assert len(keypoint_list_original) % 3 == 0

    keypoint_list_new = keypoint_list_original
    for i in range(int(len(keypoint_list_original) / 3)):
        xpos = keypoint_list_original[i * 3]
        ypos = keypoint_list_original[i * 3 + 1]
        # dont't change the third element (2)

        # Scale and update
        keypoint_list_new[i * 3] = round(xpos * width_scale, 2)
        keypoint_list_new[i * 3 + 1] = round(ypos * height_scale, 2)
        # dont't change the third element (2)

    return keypoint_list_new


def rescale_bbox(bbox_original, ori_size, new_size):
    """
    Rescales the bounding box coordinates to the correct locations in the
    resized image.

    Parameters:
        bbox_original: list of bounding box coordinates

        ori_size: tuple of ints
            The size of the original image. (Height, Width)

        new_size: tuple of ints
            The size of the new image. (Height, Width)

    Returns:
        new_bbox: list of bounding box coordinates rescaled to their locations
            on the new rescaled image.

        new_area: the new area of the rescaled bounding box
    """
    # bbox format: [top left x position, top left y position, width, height]

    # Determine scales
    height_scale, width_scale = caculate_scales(ori_size, new_size)

    # Scale top left position
    top_left_x = round(width_scale * bbox_original[0], 2)
    top_left_y = round(height_scale * bbox_original[1], 2)

    # Scale width
    width = round(width_scale * bbox_original[2], 2)
    height = round(height_scale * bbox_original[3], 2)

    # Set new bbox
    new_bbox = [top_left_x, top_left_y, width, height]

    # Determine new area
    new_area = width * height

    return new_bbox, new_area


def correct_all_json_files(
    STANDARD_SIZE, DATA_LOC: str, NEW_DATA_LOC: str, img_data_list, subdir_names=None
):
    """
    Enumerates through all subdirectorys in the dataset and corrects their JSON
    files so that annotations are rescaled by correct scaling factors.

    Parameters:
        STANDARD_SIZE: tuple of ints
            The size we are rescaling all of the images and annotations to.

        DATA_LOC: str
            The path to the dataset directory. (Parent directory of all subdirectories)

        NEW_DATA_LOC: str
            The path to the new dataset directory. (Parent directory of all subdirectories)

        img_data_list: list of img_data structures (one for each subdirectory)
            img_data_list[i] is a list of tuples (file_list, size_list) where
            file_list is a list of file names in a subdirectory and size_list
            is a list of tuples (height, width) of the images in that
            subdirectory.

        subdir_names: list of strs
            A list of the names of the subdirectories in the dataset.

    Returns:
        A list of dictionaries where each list item contains a dictionary with
        corrected subdirectory JSON annotations.
    """

    if subdir_names is None:
        subdir_names = []


    # store returns in list
    json_dict_list = []
    for i, name in enumerate(subdir_names):
        json_dict_list.append(
            correct_json_file(
                STANDARD_SIZE, DATA_LOC, NEW_DATA_LOC, img_data_list[i], name
            )
        )

    return json_dict_list


def find_json_file(DATA_LOC: str, subdir_name: str):
    """
    Finds the JSON file for the given subdirectory. Each subdirectory of a
    dataset has a JSON file containing the annotations for the images within
    that subdirectory.

    Parameters:
        DATA_LOC: str
            The path to the dataset containing directory.

        subdir_name: str
            The name of the subdirectory to find the JSON file for.
        
    Returns:
        json_filename: str
            The path to the JSON file.
    """
    # Find the json file
    json_files = glob.glob(os.path.join(DATA_LOC, subdir_name, "*.json"))

    # Assert that only one json is found
    assert len(json_files) > 0, f"No json file found in {subdir_name}"
    assert len(json_files) == 1, f"More than one json file found in {subdir_name}"

    # Return json file
    return json_files[0]


def correct_json_file(
    STANDARD_SIZE, DATA_LOC: str, NEW_DATA_LOC: str, img_data, subdir_name: str
):
    """
    Corrects the JSON file for the given subdirectory and writes this JSON
    to the appropriate file location. Each subdirectory of a dataset has a JSON
    file containing the annotations for the images within that subdirectory.

    Parameters:
        STANDARD_SIZE: tuple of ints
            The size we are rescaling all of the images and annotations to.

        DATA_LOC: str
            The path to the dataset directory. (Parent directory of all subdirectories)

        NEW_DATA_LOC: str
            The path to the new dataset directory. (Parent directory of all subdirectories)

        img_data: tuple of (file_list, size_list)
            file_list is a list of file names in a subdirectory

            size_list is a list of tuples (height, width) of the images in that
            subdirectory.

        subdir_name: str
            The name of the current subdirectory we are correcting the JSON
            file for.

    Returns:
        json_dict: dict
            The dictionary containing the corrected JSON file information.
    """

    # get json file
    json_filename = find_json_file(DATA_LOC, subdir_name)

    # Get the json_dict
    json_dict = read_in_json(json_filename)

    # Convert the img_data structure to a dict for easy access to previous sizes
    img_data_dict = {}
    create_img_data_dict(img_data, img_data_dict)

    # EDIT JSON DICT
    image_dict_list = json_dict["images"]
    annotate_dict_list = json_dict["annotations"]

    # Copy of annotate dict list to optimize search
    search_annotate_dict = create_search_annotate_dict(annotate_dict_list.copy())

    # For each image in image_dict_list
    for image_dict in image_dict_list:

        # Pull vital info from image_dict
        im_id = image_dict["id"]
        im_fn = image_dict["file_name"]

        # Pull vital info from img_data (1 0 is correct order here)
        ori_width = img_data_dict[im_fn][1]
        ori_height = img_data_dict[im_fn][0]

        # Set new width and height (0 1 is correct order here)
        image_dict["width"] = STANDARD_SIZE[0]
        image_dict["height"] = STANDARD_SIZE[1]

        # Use search dict to find annotations for this image
        annotate_dict = None
        try:
            annotate_dict = search_annotate_dict[im_id]
        except KeyError:
            pass
            # print(f"Image {im_fn} does not have annotations")

        # only correct annotations if they exist for this image
        if annotate_dict is not None:
            # First, correct bbox and area annotations (if they exist)
            bbox_original = None
            try:
                bbox_original = annotate_dict["bbox"]
            except KeyError:
                pass

            # if we have a bbox, set the new bbox and area
            if bbox_original is not None:
                # Set new bbox and area
                ori_size = (ori_width, ori_height)
                bbox_new, area_new = rescale_bbox(
                    bbox_original, ori_size, STANDARD_SIZE
                )

                # set in dict
                annotate_dict["bbox"] = bbox_new
                annotate_dict["area"] = area_new

            # Now attempt to do the same thing for the keypoints
            keypoint_list = None
            try:
                keypoint_list = annotate_dict["keypoints"]
            except KeyError:
                pass

            if keypoint_list is not None:
                new_keypoint_list = rescale_keypoints(
                    keypoint_list, ori_size, STANDARD_SIZE
                )

                annotate_dict["keypoints"] = new_keypoint_list

    # Get json_dict file name
    json_file_shortname = str.split(json_filename, os.sep)[-1]

    # Emphasize new json filename
    new_json_filename = os.path.join(NEW_DATA_LOC, subdir_name, json_file_shortname)

    # Create new json file
    json_string = json.dumps(json_dict)
    with open(new_json_filename, "w") as outfile:
        outfile.write(json_string)

    return json_dict


def read_in_json(json_filename: str):
    """
    Read in the JSON file and return the dictionary.

    Parameters:
        json_filename: str
            The path to the JSON file.

    Returns:
        json_dict: dict
            The dictionary containing the JSON file information.
    """
    with open(json_filename) as jf:
        data = json.load(jf)
        return data


def make_resized_image_directories(NEW_DATA_LOC, subdir_names=None):
    """
    Makes directories for the resized images if they do not already exist.

    Parameters:
        NEW_DATA_LOC: str
            The path to the directory where the resized images will be stored.
    
        subdir_names: list of str
            The names of the subdirectories that contain the images.
    """

    if subdir_names is None:
        subdir_names = ["train_set", "val_set", "test_set"]

    # Create these folders
    for name in subdir_names:
        folder_to_create = os.path.join(NEW_DATA_LOC, name, "images")

        try:
            os.makedirs(folder_to_create)
        except OSError:
            print(f"Folder {name} already exists")


def create_resized_images_all_subdir(stack_list, img_data_list, NEW_DATA_LOC, subdir_names=None):
    """
    Creates resized images for all subdirectories.

    Parameters:
        stack_list: List of image stacks (each Nx3xHxW) where each stack is a 
            different subdirectory of images.
        
        img_data_list: list of img_data structures (one for each subdirectory)
            img_data_list[i] is a list of tuples (file_list, size_list) where
            file_list is a list of file names in a subdirectory and size_list
            is a list of tuples (height, width) of the images in that
            subdirectory.
        
        NEW_DATA_LOC: location to save resized subdirectories
        
        subdir_names: list of subdirectories to be resized
            Default: ["train_set", "val_set", "test_set"]
    """

    if subdir_names is None:
        subdir_names = ["train_set", "val_set", "test_set"]

    # Create these folders
    for i, name in enumerate(subdir_names):
        create_resized_images_for_subdir(stack_list[i], img_data_list[i], NEW_DATA_LOC, name)


def create_resized_images_for_subdir(images, img_data, NEW_DATA_LOC, name):
    """
    Creates resized images of the images inside the subdir images folder

    Parameters:
        images: list of images to be resized

        img_data: list of img_data structures

        NEW_DATA_LOC: location of new data folder

        name: name of the subdirectory to create a resized images folder for
    """
    # Keep track of how many images saved successfully and which ones did not
    success_count = 0

    # Keeps track of which images
    failures = []

    # Unpack img_data
    file_list, size_list = img_data

    # Make the label directories, if they do not exist
    make_resized_image_directories(NEW_DATA_LOC, [name])

    # For each image
    for i in range(images.shape[0]):
        # Get the current file name
        f = file_list[i]

        # Determine the name of this image
        ext_len = len(str.split(str.split(f, os.sep)[-1], ".")[-1])

        file_name = str.split(f, os.sep)[-1][0 : -(ext_len + 1)]

        # Determine which set the image is in
        image_set = str.split(f, os.sep)[-3]

        # Create final image path
        image_path = os.path.join(NEW_DATA_LOC, image_set, "images", file_name + ".jpg")

        # Write image to path location
        if cv2.imwrite(image_path, images[i]):
            success_count += 1
        else:
            failures.append(f)

    if len(failures) > 0:
        print("Images that failed to generate resized images:")
        for fail in failures:
            print(fail)

    print("")
    print(f"Successfully saved {success_count} resized images for {name}")
    print("")


def read_in_all_images(DATA_LOC: str, STANDARD_SIZE, subdir_names = None):
    """
    Iterates through all subdirectories in a specified dataset and reads in all
    images.

    Parameters:
        DATA_LOC: str
            The path to the dataset.
    
        STANDARD_SIZE: tuple of ints
            The size all images will be resized to in the dataset.

        subdir_names: list of str
            The names of the subdirectories that contain the images.

    Returns:
        stack_list: list of image stacks (each Nx3xHxW) where each stack is a
            different subdirectory of images.

        img_data_list: list of img_data structures (one for each subdirectory)
            img_data_list[i] is a list of tuples (file_list, size_list) where
            file_list is a list of file names in a subdirectory and size_list
            is a list of tuples (height, width) of the images in that
            subdirectory.
    """
    if subdir_names is None:
        subdir_names = ["train_set", "val_set", "test_set"]

    stack_list = []
    img_data_list = []
    for name in subdir_names:
        stack, img_data = read_in_subdir_images(DATA_LOC, STANDARD_SIZE, name)
        stack_list.append(stack)
        img_data_list.append(img_data)

    return stack_list, img_data_list


def read_in_subdir_images(DATA_LOC: str, STANDARD_SIZE, name):
    """
    Reads in images from the subdirectory and returns a numpy array containing
    the images of shape (NxHxWxC), where N is the number of images, C is the
    number of channels, and H + W are sizes of spatial dimensions.

    Parameters:
        DATA_LOC: str
            The path to the dataset.

        STANDARD_SIZE: tuple of int
            The size all images will be resized to in the dataset

        name: str
            The name of the subdirectory to read in images from
    """

    # Glob the data together
    x = []

    # Collect all *.jpg files in sub-directory 'name'
    file_list = glob.glob(os.path.join(DATA_LOC, name, "images", "*"))

    # save the sizes of each file so we can resize at the end
    size_list = []
    print("Loading in and resizing images from {name}:")
    print(f"{len(file_list)} image files detected")
    for f in file_list:
        image = cv2.imread(f)
        size_list.append((image.shape[0], image.shape[1]))
        x.append(np.array(cv2.resize(image, (STANDARD_SIZE[0], STANDARD_SIZE[1]))))

    # Stack images into 4D NxHxWxC
    stack = np.array(x)

    print()
    print("Images loaded and resized successfully")
    print(f"Returned stack has size NxHxWxC: {stack.shape}")
    return stack, (file_list, size_list)
