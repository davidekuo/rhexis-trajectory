"""
A module of utility functions for the Rhexis project Keypoint Detection
functionality.
"""
import os
import glob
import json
import argparse
import cv2, random
from matplotlib import pyplot as plt

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

from detectron2.config import get_cfg
from detectron2 import model_zoo


def load_cfg(
    model_string: str,
    MAX_ITER=1500,
    BASE_LR=0.000025,
    OKS_SIGMAS = None,
    SOLVER_STEPS = None,
):
    """
    Returns the config object for the specified model

    Parameters:
        model_string(str): A string specifying which model we would like to use

        MAX_ITER(int): The number of iterations to train for
            - Note: This is not the number of epochs. If the model has already
            been trained, the previously trained iterations will count toward
            this number.
            -Default: 1500
        
        BASE_LR(float): The base learning rate for the model
            -Default: 0.000025

        OKS_SIGMAS(list): A list of two floats specifying the sigma values for
            the OKS loss.
            -Default: [0.03, 0.03]

        SOLVER_STEPS(list): A list of integers specifying the steps at which
            the learning rate should decay.
            -Default: []
        
    Returns:
        cfg(dict): The config object for the specified model
    """
    if SOLVER_STEPS is None:
        SOLVER_STEPS = []

    if OKS_SIGMAS is None:
        OKS_SIGMAS = [0.03, 0.03]
    
    cfg = get_cfg()

    model_string = model_string + ".yaml"

    valid_options = [
        "keypoint_rcnn_R_50_FPN_3x.yaml",
        "keypoint_rcnn_R_101_FPN_3x.yaml",
        "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml",
    ]

    if model_string not in valid_options:
        assert False

    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Keypoints/{model_string}"))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.TEST.EVAL_PERIOD = 100

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        f"COCO-Keypoints/{model_string}"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = BASE_LR  # pick a good LR
    cfg.SOLVER.MAX_ITER = MAX_ITER  # 300 was good for balloon toy dataset. Adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = SOLVER_STEPS  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for balloon toy dataset (default: 512)
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (utrada tip). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = OKS_SIGMAS

    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    # tells Detectron2 to save a model checkpoint every X iterations
    # checkpoints are saved as 'model_{iteration_number}.pth
    # Detectron2 also creates a file 'last_checkpoint' which simply contains the filename of .pth file for the last checkpoint (ex. model_0000079.pth)
    # To resume training from last_checkpoint, Detectron2 needs 'last_checkpoint' and the corresponding .pth file in cfg.OUTPUT_DIR

    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # keep and do not exclude images labeled to have no objects

    return cfg


def create_experiment_name(
    model_string: str,
    BASE_LR=0.000025,
    OKS_SIGMAS=None,
    SOLVER_STEPS=None,
    augmentation=False,
):
    """
    Creates a name for the experiment based on the specified model and 
    parameters.
    
    Parameters:
        model_string(str): A string specifying which model we would like to use

        BASE_LR(float): The base learning rate for the model
            -Default: 0.000025

        OKS_SIGMAS(list): A list of two floats specifying the sigma values for
            the OKS loss.
            -Default: [0.03, 0.03]

        SOLVER_STEPS(list): A list of integers specifying the steps at which
            the learning rate should decay.
            -Default: []
        
        augmentation(bool): A boolean specifying whether or not we are using
            augmentation.
            -Default: False

    Returns: 
        A string specifying the name of the experiment
    """
    if SOLVER_STEPS is None:
        SOLVER_STEPS = []

    if OKS_SIGMAS is None:
        OKS_SIGMAS = [0.03, 0.03]

    name = model_string + "_"
    name = name + f"LR_{BASE_LR}" + "_"
    name = name + f"OKS_SIGMAS_{OKS_SIGMAS[0]},{OKS_SIGMAS[1]}" + "_"
    if not SOLVER_STEPS:
        name = name + "LRDECAY_NONE"
    else:
        name = name + f"LRDECAY_ACTIVE-FirstAt{SOLVER_STEPS[0]}_"
    if augmentation:
        name = name + "augmentation_YES"
    else:
        name = name + "augmentation_NO"

    return name


def get_dataset_location(dataset_name: str, DATASET_LOC: str):
    """
    Returns the location of the the specified dataset's JSON file and image
    directory.

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
    Loads and does minor preprocessing on the datasets.
    After this function is called.  You can now use:
    1. MetadataCatalog
    2. DatasetCatalog
    with "train", "test" and "val" for the datset names

    Parameters:
        DATASET_LOC(str): Path to the directory containing the datasets
    """
    rhexis_keypoint_names = ["utrada_tip1", "utrada_tip2"]
    rhexis_flip_map = [("utrada_tip1", "utrada_tip2")]

    for dataset in ["train", "test", "val"]:
        # Collect json and image location
        dataset_json_loc, dataset_image_loc = get_dataset_location(dataset, DATASET_LOC)
        register_coco_instances(dataset, {}, dataset_json_loc, dataset_image_loc)

        # add keypoint_names metadata needed for training
        MetadataCatalog.get(dataset).keypoint_names = rhexis_keypoint_names
        MetadataCatalog.get(dataset).keypoint_flip_map = rhexis_flip_map


def visualize_image_annotations(dataset: str, n_images: int = 5):
    """
    Visualizes the annotations of the specified dataset by displaying <n_images>
    random images with their annotations.
    
    Parameters:
        dataset(str): Name of the dataset to visualize
        
        n_images(int): Number of images to visualize
    """

    # visualize data
    dataset_dicts = DatasetCatalog.get(dataset)
    metadata = MetadataCatalog.get(dataset)

    for d in random.sample(dataset_dicts, n_images):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image())
        plt.show()


def best_model_checkpoint(output_dir: str) -> str:
    """
    Evaluates the file metrics.json in the Detectron2 temporary output
    directory to determine the saved checkpoint with the highest keypoint AP.
    Returns the string path of the best checkpoint .pth file.

    Parameters:
      output_dir: string containing path to output directory

    Returns:
      max_kp_AP_checpoint: string of filename of best-performing checkpoint
    """

    with open(output_dir + os.sep + "metrics.json") as read_file:
        lines = read_file.readlines()

    checkpoint_list = glob.glob(os.path.join(output_dir, "*.pth"))

    max_kp_AP = 0
    max_kp_AP_checkpoint = ""

    for line in lines:
        json_dict = json.loads(line)
        if "keypoints/AP" in json_dict:
            kp_AP = json_dict["keypoints/AP"]
            iteration = json_dict["iteration"]
            checkpoint = "model_" + f"{iteration}".zfill(7) + ".pth"
            res = [i for i in checkpoint_list if checkpoint in i]
            if kp_AP > max_kp_AP and len(res) == 1:
                max_kp_AP = kp_AP
                max_kp_AP_checkpoint = checkpoint

    return max_kp_AP_checkpoint