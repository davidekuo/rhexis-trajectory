"""
A module of utility functions for the Rhexis project.
"""

from typing_extensions import assert_type
import os

import cv2, random
from matplotlib import pyplot as plt

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def get_json_and_images(dataset_name: str, DATASET_LOC: str):
    """
    This function returns the location of the the specified dataset.

    Parameters:
        dataset_name(str): A string specifying which dataset we would like to
        collect the json and image locations of.
        - 'train' to collect training dataset
        - 'val' to collect validation set
        - 'test' to collect test set

    Returns:
        json_loc(str): File path to JSON keypoint annotations
        image_loc(str): File path to images directory


    """

    # Ensure the input is the correct type
    assert_type(dataset_name, str)
    assert_type(DATASET_LOC, str)

    # Ensure this input is one of the valid options
    msg = f"'{dataset_name}' is an invalid option: Please use 'train', 'val', or 'test'"
    assert (
        dataset_name == "train" or dataset_name == "val" or dataset_name == "test"
    ), msg

    # Return the locations
    json_loc = os.path.join(
        DATASET_LOC, f"{dataset_name}_set", f"{dataset_name}_coco_annotations.json"
    )

    image_loc = os.path.join(DATASET_LOC, f"{dataset_name}_set", "images")

    return json_loc, image_loc


def load_datasets_pipeline(DATASET_LOC: str):
    """
    This loads and does minor preprocessing on the datasets.
    After this function is called.  You can now use:
    1. MetadataCatalog
    2. DatasetCatalog
    with "train", "test" and "val" for the datset names
    """
    rhexis_keypoint_names = ["utrada_tip1", "utrada_tip2"]
    rhexis_flip_map = [("utrada_tip1", "utrada_tip2")]

    for dataset in ["train", "test", "val"]:
        # Collect json and image location
        dataset_json_loc, dataset_image_loc = get_json_and_images(dataset, DATASET_LOC)
        register_coco_instances(dataset, {}, dataset_json_loc, dataset_image_loc)

        # add keypoint_names metadata needed for training
        MetadataCatalog.get(dataset).keypoint_names = rhexis_keypoint_names
        MetadataCatalog.get(dataset).keypoint_flip_map = rhexis_flip_map


def visualize_image_annotations(dataset, n_images):
    # visualize data
    dataset_dicts = DatasetCatalog.get(dataset)
    metadata = MetadataCatalog.get(dataset)

    for d in random.sample(dataset_dicts, n_images):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image())
        plt.show()
