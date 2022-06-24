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


def convert_bbox(bbox):
    """
    Converts a bounding box from the output of the model to the standard COCO
    format for bounding boxes. Note: Assumes 0 indexing from top left corner for
    both x and y.
    
    Parameters:
        bbox(list): A list of four floats representing the bounding box
            in the format that the model outputs.
            [<left top x>, <left top y>, <right bottom x>, <right bottom y>]
            
    Returns:
        bbox(list): A list of four floats representing the bounding box in the
            standard COCO format.
            [<left top x>, <left top y>, <width>, <height>]
    """
    x0 = bbox[0]
    y0 = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [x0, y0, width, height]


class RhexisPredictor:
    """
    A class for predicting the bounding boxes and keypoints using the loaded
    model.

    Attributes:
        predictor(torch.nn.Module): The loaded model.
    """
    def __init__(self, model_weights_path, cfg, object_detection_threshold=0.98):
        """
        Initializes the RhexisPredictor class.
        
        Parameters:
            model_weights_path(str): The path to the model weights.

            cfg(dict): The loaded configuration file

            object_detection_threshold(float): The threshold for object
                detection.
                Default: 0.98
                
            Raises:
                FileNotFoundError: If the model weights file is not found.
        """
        cfg.MODEL_WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = object_detection_threshold
        self.predictor = DefaultPredictor(cfg)

    def randomly_sample_images_from_set(self, set_to_sample="test", num_images=5):
        """
        Samples <num_images> images from the specified set.

        Parameters:
            set_to_sample(str): The set to sample images from.
                Options: "train", "val", "test"
                Default: "test"
            
            num_images(int): The number of images to sample.
                Default: 5
        
        Returns:
            images(list): A list of dictionaries containing the images
        """
        dataset_dicts = DatasetCatalog.get(set_to_sample)
        return random.sample(dataset_dicts, num_images)

    def detectron2_visualizer(
        self, set_to_sample="test", num_images=5, sampled_list=None
    ):
        """
        Visualizes the predictions for the specified set by displaying the images
        with overlayed bounding boxes and keypoints.

        Parameters:
            set_to_sample(str): The set to sample images from.
                Options: "train", "val", "test"
                Default: "test"
        
            num_images(int): The number of images to sample.
                Default: 5

            sampled_list(list): A list of dictionaries containing the images
                to visualize. If this is None, then the images will be sampled
                from the specified set.
                Default: None
        """
        list_of_file_dicts = sampled_list
        if sampled_list is None:
            print(f"Randomly sampling {num_images} images from {set_to_sample} set")
            list_of_file_dicts = self.randomly_sample_images_from_set(
                set_to_sample, num_images
            )
        else:
            print(f"Showing {len(sampled_list)} files from past in list")

        for d in list_of_file_dicts:
            im = cv2.imread(d["file_name"])
            outputs = self.predictor(
                im
            )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            v = Visualizer(
                im[:, :, ::-1], metadata=MetadataCatalog.get("train"), scale=0.5
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2_imshow(out.get_image()[:, :, ::-1])

    def predict_keypoint_features(self, im):
        """
        Predicts the keypoint features for the image.
        
        Parameters:
            im(np.ndarray): The image to predict the keypoint features for.
            
        Returns:
            keypoints(list): A list of floats representing the keypoints in the
                COCO format for keypoints.
                [<x1>, <y1>, 2, <x2>, <y2>, 2, ...]
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.predictor(im)

    def compare_labels(self, set_to_sample="test", num_images=5, sampled_list=None):
        """
        Compares the ground truth and predicted labels for a set of sampled images,
        or explicitly referenced images.
        
        Parameters:
            set_to_sample(str): The set to sample images from.
                Options: "train", "val", "test"
                Default: "test"

            num_images(int): The number of images to sample.
                Default: 5
            
            sampled_list(list): A list of dictionaries containing the images
                to visualize. If this is None, then the images will be sampled
                from the specified set.
                Default: None
        """

        list_of_file_dicts = sampled_list
        if sampled_list is None:
            print(f"Randomly sampling {num_images} images from {set_to_sample} set")
            list_of_file_dicts = self.randomly_sample_images_from_set(
                set_to_sample, num_images
            )
        else:
            print(f"Showing {len(sampled_list)} files from past in list")

        for d in list_of_file_dicts:
            im = cv2.imread(d["file_name"])
            print("=========")
            print(f"File: {str.split(d['file_name'],os.sep)[-1]}")

            # Get the ground truth label data
            gt_bbox = d["annotations"][0]["bbox"]

            # Get the ground truth keypoints label data
            gt_keypoint = d["annotations"][0]["keypoints"]

            # Calculate predictions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = self.predictor(im)

            pr_bbox = self.get_bbox_prediction(outputs)[0]

            pr_keypoints = self.get_keypoint_prediction(outputs)[0]

            print("Ground Truth is green")
            print("Predictions are blue")
            if gt_bbox is not None and pr_bbox is not None:
                im = compare_bboxes(im, gt_bbox, pr_bbox)

            if gt_keypoint is not None and pr_keypoints is not None:
                im = compare_keypoints(im, gt_keypoint, pr_keypoints)

            cv2_imshow(im)

    def get_bbox_prediction(self, outputs):
        """
        Gets the bounding box prediction from the output of the keypoint
        dection model.

        Parameters:
            outputs(dict): The output of the keypoint detection model.
                See the documentation for the keypoint detection model at
                https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        Returns:
            bbox_prediction(list): A list of floats representing the bounding
                boxes in the COCO format for bounding boxes.
                [<x1>, <y1>, 2, <x2>, <y2>, 2, ...]
        """
        try:
            # remove this print when done with dev
            # print_dict(outputs)
            pr_bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            bbox_list = []
            for bbox in pr_bboxes:
                bbox_list.append(convert_bbox(bbox))
        except:
            bbox_list = None
        return bbox_list

    def get_keypoint_prediction(self, outputs):
        """
        Gets the keypoint prediction from the output of the keypoint
        dection model.

        Parameters:
            outputs(dict): The output of the keypoint detection model.
                See the documentation for the keypoint detection model at
                https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        Returns:
            keypoint_prediction(list): A list of floats representing the keypoints
                in the following format for keypoints.
                [<x1>, <y1>, 2, <x2>, <y2>, 2, ...]
        """
        try:
            pr_keypoints_list = outputs["instances"].pred_keypoints.cpu().numpy()
            keypoint_list = []
            for keypoints in pr_keypoints_list:
                keypoint_list.append(keypoints.flatten())
        except:
            keypoint_list = None
        return keypoint_list



