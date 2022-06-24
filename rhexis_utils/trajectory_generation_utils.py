"""
Functions that generate trajectories
"""
from operator import le
import os
import sys
from scipy import spatial

from torch.nn.functional import fractional_max_pool2d

sys.path.insert(0, "./Feature_Extraction")
sys.path.insert(0, ".Semantic_Segmentation")
sys.path.insert(0, "./Keypoint_Detection")
import label_feature_extraction as lfe
import segmentation_access_functions as saf
from eval_model import RhexisVisualizer
import glob
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import cv2
from detectron2.config import get_cfg
from google.colab.patches import cv2_imshow
import numpy as np


def generate_trajectories(
    DATA_LOC: str,
    subdir_list,
    BEST_MODEL_CHECKPOINT,
    use_image_folder=False,
    MANUAL_ANNOTATION_LOC=None,
):
    # For each subdir in the list, we need to generate a seperate csv dict
    for f, subdir in enumerate(subdir_list):
        print(f"Folder {f} of {len(subdir_list)}")
        generate_csv_dict(
            DATA_LOC,
            subdir,
            BEST_MODEL_CHECKPOINT,
            use_image_folder,
            MANUAL_ANNOTATION_LOC,
        )


def intalize_csv_dict(csv_dict):
    # FRAME NUM | IMAGE_FILENAME | KeyL_X | KeyL_Y | KeyR_X | KeyR_Y | BBOX | Pupil Center | Pupil Extents x4 | Incision |
    csv_dict.setdefault("frame_num", [])
    csv_dict.setdefault("image_path", [])
    csv_dict.setdefault("label_path", [])
    csv_dict.setdefault("key_L_x", [])
    csv_dict.setdefault("key_L_y", [])
    csv_dict.setdefault("key_R_x", [])
    csv_dict.setdefault("key_R_y", [])
    csv_dict.setdefault("bbox_x", [])
    csv_dict.setdefault("bbox_y", [])
    csv_dict.setdefault("bbox_width", [])
    csv_dict.setdefault("bbox_height", [])
    csv_dict.setdefault("pupil_center_x", [])
    csv_dict.setdefault("pupil_center_y", [])
    csv_dict.setdefault("pupil_left_x", [])
    csv_dict.setdefault("pupil_right_x", [])
    csv_dict.setdefault("pupil_up_y", [])
    csv_dict.setdefault("pupil_down_y", [])
    csv_dict.setdefault("incision_x", [])
    csv_dict.setdefault("incision_y", [])


def generate_csv_dict(
    DATA_LOC,
    subdir,
    BEST_MODEL_CHECKPOINT,
    use_image_folder=True,
    MANUAL_ANNOTATION_LOC=None,
):
    print(f"Generating features.csv for {subdir}")

    # Set up keypoint prediction model
    yaml_filepath = os.sep + os.path.join(
        *str.split(BEST_MODEL_CHECKPOINT, os.sep)[0:-1], "config.yaml"
    )
    print(yaml_filepath)
    cfg = None
    # with open(yaml_filepath) as f:
    #  cfg = yaml.load(f, Loader = SafeLoader)

    cfg = get_cfg()
    cfg.merge_from_file(yaml_filepath)
    cfg.merge_from_list(["MODEL.WEIGHTS", BEST_MODEL_CHECKPOINT])

    # Set up RhexisVisualizer object
    RV = RhexisVisualizer(BEST_MODEL_CHECKPOINT, cfg)

    # Set up label dict
    label_dict = saf.get_labels(task=2)

    # Get a list of frames in this subdir
    frame_list = []
    if use_image_folder:
        frame_list = glob.glob(os.path.join(DATA_LOC, subdir, "images", "*.jpg"))
    else:
        frame_list = glob.glob(os.path.join(DATA_LOC, subdir, "*.jpg"))

    # Finally, determine if this folder has manually labeled trajectories
    manual_df = None
    if MANUAL_ANNOTATION_LOC is not None:
        if subdir.startswith("AC1"):
            manual_df = pd.read_csv(os.path.join(MANUAL_ANNOTATION_LOC, "pgy4.csv"))
        elif subdir.startswith("CataractCoach1"):
            manual_df = pd.read_csv(os.path.join(MANUAL_ANNOTATION_LOC, "expert.csv"))

    # Intialize the dict
    csv_dict = {}
    intalize_csv_dict(csv_dict)
    i = -1
    for frame in frame_list:
        i += 1

        if i % 20 == 0:
            print(f"- Starting frame {i} / {len(frame_list)}")

        # Append frame number
        extension_len = len(str.split(frame, ".")[-1])
        frame_num = int(str.split(frame, "_")[-1][0 : -extension_len - 1])

        csv_dict["frame_num"].append(frame_num)

        # Append image_path
        csv_dict["image_path"].append(frame)

        # Append label_path
        filename = str.split(frame, os.sep)[-1]
        label_path = os.path.join(DATA_LOC, subdir, "labels", filename)[0:-4] + ".png"
        csv_dict["label_path"].append(label_path)

        # Open the image file
        im = cv2.imread(frame)
        label = cv2.imread(label_path)[:, :, 0]

        # SEGMENTATION FEATURES
        # Generate the filled pupil once and use for all of the remaining features
        # for optimization
        full_pupil = lfe.extract_pupil_filled(label, label_dict)
        # Extract pupil center
        pupil_median = lfe.extract_pupil_median_pos(label, label_dict, full_pupil)
        # Numpy uses y x coords
        p_medy, p_medx = pupil_median
        csv_dict["pupil_center_x"].append(p_medx)
        csv_dict["pupil_center_y"].append(p_medy)

        # Extract pupil extents
        extents = lfe.extract_pupil_extents(label, label_dict, full_pupil, pupil_median)
        l_ex, r_ex, u_ex, d_ex = extents
        # Numpy uses y x coords
        csv_dict["pupil_left_x"].append(l_ex[1])
        csv_dict["pupil_right_x"].append(r_ex[1])
        csv_dict["pupil_up_y"].append(u_ex[0])
        csv_dict["pupil_down_y"].append(d_ex[0])

        # Extract incision position
        in_x, in_y = lfe.extract_incision_position(label, label_dict)
        csv_dict["incision_x"].append(in_x)
        csv_dict["incision_y"].append(in_y)

        # KEYPOINT PREDICTIONS
        # Run our model on this image for keypoint prediction
        predictions = RV.predict_keypoint_features(im)

        # Get the bbox predictions
        bbox = RV.get_bbox_prediction(predictions)

        # Get the keypoint predictions
        keypoint_pairs = None
        if manual_df is None:
            keypoint_pairs = RV.get_keypoint_prediction(predictions)
        else:
            keypoint_pairs = get_keypoint_label(frame_num, manual_df)
            if keypoint_pairs is None:
                bbox = []

        # Determine which keypoints and bounding boxes are correct
        if len(bbox) == 0:
            append_nans_for_bbox_and_keys(csv_dict)
        else:
            # lfe.display_dot_on_pos(lfe.display_dot_on_pos(label, extents), pupil_median)
            correct_keypoint_i = determine_correct_keypoints(
                keypoint_pairs, pupil_median, extents
            )

            if correct_keypoint_i == -1:
                append_nans_for_bbox_and_keys(csv_dict)
            else:
                # Collect the correct bbox and keypoints
                bb = bbox[correct_keypoint_i]
                key = keypoint_pairs[correct_keypoint_i]

                # Add these values to the csv_dict
                append_bb_and_key(csv_dict, bb, key)

    # Write the file
    write_dict_to_csv(csv_dict, subdir, DATA_LOC, manual_df)


def get_keypoint_label(frame_num, manual_df):
    q = manual_df.loc[manual_df["frame"] == frame_num]

    if not q.empty:
        a = q.to_numpy()
        X = abs(a[0][1])
        Y = abs(a[0][2])
        return [[X, Y, 2, X, Y, 2]]
    else:
        return None


def append_bb_and_key(csv_dict, bbs, key):
    # add bbox coords
    csv_dict["bbox_x"].append(bbs[0])
    csv_dict["bbox_y"].append(bbs[1])
    csv_dict["bbox_width"].append(bbs[2])
    csv_dict["bbox_height"].append(bbs[3])

    # Determine which of the key points is L and which is R
    keypointA_x = key[0]
    keypointB_x = key[3]

    if keypointA_x < keypointB_x:
        L = (keypointA_x, key[1])
        R = (keypointB_x, key[4])
    else:
        R = (keypointA_x, key[1])
        L = (keypointB_x, key[4])

    csv_dict["key_L_x"].append(L[0])
    csv_dict["key_L_y"].append(L[1])
    csv_dict["key_R_x"].append(R[0])
    csv_dict["key_R_y"].append(R[1])


def append_nans_for_bbox_and_keys(csv_dict):
    # If there are no bbox/ keypoints detected, return NA
    csv_dict["key_L_x"].append(np.nan)
    csv_dict["key_L_y"].append(np.nan)
    csv_dict["key_R_x"].append(np.nan)
    csv_dict["key_R_y"].append(np.nan)
    csv_dict["bbox_x"].append(np.nan)
    csv_dict["bbox_y"].append(np.nan)
    csv_dict["bbox_width"].append(np.nan)
    csv_dict["bbox_height"].append(np.nan)


def determine_correct_keypoints(keypoint_pairs, median, extents):
    # Unpack the keypoints
    keypoints = [item for sublist in keypoint_pairs for item in sublist]

    # unpack extents NOTE: Numpy uses y x coords
    l_ex = extents[0][1]
    r_ex = extents[1][1]
    u_ex = extents[2][0]
    d_ex = extents[3][0]

    # we only want to keep keypoints that are inside the pupil mask

    # Create list of indicies that assign values to their respective keypoint pairs
    ind = np.array(range(len(keypoints)))
    ind = np.floor(ind / 6)

    x_pos_list = [x for i, x in enumerate(keypoints) if i % 3 == 0]
    y_pos_list = [y for i, y in enumerate(keypoints) if i % 3 == 1]
    ind_list = [index for i, index in enumerate(ind) if i % 3 == 0]

    final_x_list = []
    final_y_list = []
    final_ind_list = []
    for i in range(len(ind_list)):
        # Only keep keypoints inside the pupil
        # Must be inside x
        if x_pos_list[i] > l_ex and x_pos_list[i] < r_ex:
            if y_pos_list[i] < d_ex and y_pos_list[i] > u_ex:
                final_x_list.append(x_pos_list[i])
                final_y_list.append(y_pos_list[i])
                final_ind_list.append(ind_list[i])

    # At this point, if all keypoint pairs have been thrown out, return -1
    if len(final_ind_list) == 0:
        return -1

    # if only one keypoint passed, return it's index
    if len(final_ind_list) == 1:
        # Only one keypoint pair passed: one passed one didn't from pair
        return int(final_ind_list[0])

    # if two keypoints passed AND they are in the same pair, return their index
    if len(final_ind_list) == 2 and final_ind_list[0] == final_ind_list[1]:
        # Only one keypoint pair passed: both
        return int(final_ind_list[0])

    # Multiple keypoints from different bboxes
    # otherwise, we need to tie break
    # take closest point to the median (using L1 distance)
    dist = []
    for i in range(len(final_ind_list)):
        L1 = np.abs(final_x_list[i] - median[1]) + np.abs(final_y_list[i] - median[0])
        dist.append(L1)

    min_ind = np.argmin(dist)
    # Determine closest x_pos's index and return
    return int(final_ind_list[min_ind])


def write_dict_to_csv(csv_dict, subdir, DATA_LOC, manual_df=None):
    # Attempt to make the output directory
    output_folder_path = os.path.join(DATA_LOC, "OUTPUT")

    try:
        os.makedirs(output_folder_path)
    except FileExistsError:
        print("Output directory already exists")

    # generate file output location
    if manual_df is None:
        filename = subdir + "_features.csv"
    else:
        filename = subdir + "_feature_MANUAL.csv"
    print(f"Saving {filename}")
    output_file_path = os.path.join(output_folder_path, filename)

    # Generate output file path
    pd.DataFrame.from_dict(csv_dict).to_csv(output_file_path)

    print("Save complete!")
