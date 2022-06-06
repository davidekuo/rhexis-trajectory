import cv2
import os
import pathlib
import json
import torch
from torchvision.transforms import (
    ToPILImage,
    ColorJitter,
    ToTensor,
    Normalize,
    RandomApply,
)
from utils import *
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

"""
Note: This file contains some modified code from:
https://github.com/RViMLab/MICCAI2021_Cataract_semantic_segmentation/blob/b24ea3b03e80b82c6633bec6c95dcc59c704cc76/utils/utils.py
"""


def remap_experiment(mask, experiment):
    """Remap mask for Experiment 'experiment' (needs to be int)"""
    colormap = get_remapped_colormap(CLASS_INFO[experiment][0])
    remapped_mask = remap_mask(mask, class_remapping=CLASS_INFO[experiment][0])
    return remapped_mask, CLASS_INFO[experiment][1], colormap


def remap_mask(mask, class_remapping, ignore_label=255, to_network=None):
    """
    Remaps mask class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param class_remapping: dictionary that indicates class remapping
    :param ignore_label: class ids to be ignored
    :param to_network: default False. If true, the ignore value (255) is remapped to the correct number for exp 2 or 3
    :return: 2D/3D ndarray of remapped segmentation mask
    """
    to_network = False if to_network is None else to_network
    classes = []
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    n = max(len(classes), mask.max() + 1)
    remap_array = np.full(n, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    mask_remapped = remap_array[mask]
    if to_network:
        mask_remapped[mask_remapped == 255] = len(class_remapping) - 1
    return mask_remapped


def get_remapped_colormap(class_remapping):
    """
    Generated colormap of remapped classes
    Classes that are not remapped are indicated by the same color across all experiments
    :param class_remapping: dictionary that indicates class remapping
    :return: 2D ndarray of rgb colors for remapped colormap
    """
    colormap = get_cadis_colormap()
    remapped_colormap = {}
    for key, val in class_remapping.items():
        if key == 255:
            remapped_colormap.update({key: [0, 0, 0]})
        else:
            remapped_colormap.update({key: colormap[val[0]]})
    return remapped_colormap


def get_cadis_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    c_map = np.asarray(
        [
            [0, 137, 255],
            [255, 165, 0],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
            [170, 124, 0],
            [188, 255, 0],
            [0, 207, 255],
            [0, 255, 207],
            [188, 0, 255],
            [243, 0, 255],
            [0, 203, 108],
            [252, 255, 0],
            [93, 182, 177],
            [0, 81, 203],
            [211, 183, 120],
            [231, 203, 0],
            [0, 124, 255],
            [10, 91, 44],
            [2, 0, 60],
            [0, 144, 2],
            [133, 59, 59],
        ]
    )
    return c_map


def mask_from_network(mask, experiment):
    """
    Converts the segmentation masks as used in the network to using the IDs as used by the CaDISv2 paper
    :param mask: Input mask with classes numbered strictly from 0 to num_classes-1
    :param experiment: Experiment number
    :return: Mask with classes numbered as required by CaDISv2 for the specific experiment (includes '255')
    """
    if experiment == 2 or experiment == 3:
        mask[mask == len(CLASS_INFO[experiment][1]) - 1] = 255
    return mask


def mask_to_colormap(mask, colormap, from_network=None, experiment=None):
    """
    Genarates RGB mask colormap from mask with class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param colormap: dictionary that indicates color corresponding to each class
    :param from_network: Default False. If True, class IDs as used in the network are first corrected to CaDISv2 usage
    :param experiment: Needed if from_network = True to determine which IDs need to be corrected
    :return: 3D ndarray Generated RGB mask
    """
    from_network = False if from_network is None else from_network
    if from_network:
        mask = mask_from_network(mask, experiment)
    rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    # TODO: I feel this can be vectorised for speed
    for label, color in colormap.items():
        rgb[mask == label] = color
    return rgb


def plot_images(remapped_mask, remapped_colormap, classes_exp, img=None):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param img: 3D ndarray of input image
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param remapped_colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: dictionary of classes names and corresponding class ids
    :return: plot of image and rgb mask with class colorbar
    """
    mask_rgb = mask_to_colormap(remapped_mask, colormap=remapped_colormap)
    if img is not None:
        # Ensure img and label have same spatial dims
        if remapped_mask.shape != img.shape[0:2]:
            print("WARNING: Image and label dimensions do not match")
            print(f"Image shape: {img.shape}")
            print(f"Label shape: {remapped_mask.shape}")

        fig, axs = plt.subplots(1, 2, figsize=(26, 7))
        plt.subplots_adjust(
            left=1 / 16.0, right=1 - 1 / 16.0, bottom=1 / 8.0, top=1 - 1 / 8.0
        )

        # We must swap from BGR to RGB because we utilized cv2.read to read in files
        red = img[:, :, 2].copy()
        blue = img[:, :, 0].copy()
        img[:, :, 0] = red
        img[:, :, 2] = blue
        axs[0].imshow(img)
        axs[0].axis("off")

        img_u_labels = np.unique(remapped_mask)
        c_map = []
        cl = []
        for i_label in img_u_labels:
            for i_key, i_color in remapped_colormap.items():
                if i_label == i_key:
                    c_map.append(i_color)
            for i_key, i_class in classes_exp.items():
                if i_label == i_key:
                    cl.append(i_class)
        cl = np.asarray(cl)
        cmp = np.asarray(c_map) / 255
        cmap_mask = LinearSegmentedColormap.from_list(
            "seg_mask_colormap", cmp, N=len(cmp)
        )
        im = axs[1].imshow(mask_rgb, cmap=cmap_mask)
        intervals = np.linspace(0, 255, num=len(cl) + 1)
        ticks = intervals + int((intervals[1] - intervals[0]) / 2)
        divider = make_axes_locatable(axs[1])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(mappable=im, cax=cax1, ticks=ticks, orientation="vertical")
        cbar1.ax.set_yticklabels(cl)
        axs[1].axis("off")
        fig.tight_layout()

        return fig
    else:
        # Image is none
        fig = plt.figure()
        fig.set_dpi(200)
        img_u_labels = np.unique(remapped_mask)
        c_map = []
        cl = []
        for i_label in img_u_labels:
            for i_key, i_color in remapped_colormap.items():
                if i_label == i_key:
                    c_map.append(i_color)
            for i_key, i_class in classes_exp.items():
                if i_label == i_key:
                    cl.append(i_class)
        cl = np.asarray(cl)
        cmp = np.asarray(c_map) / 255
        cmap_mask = LinearSegmentedColormap.from_list(
            "seg_mask_colormap", cmp, N=len(cmp)
        )
        im = plt.imshow(mask_rgb, cmap=cmap_mask)
        intervals = np.linspace(0, 255, num=len(cl) + 1)
        ticks = intervals + int((intervals[1] - intervals[0]) / 2)
        divider = make_axes_locatable(fig.axes[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(mappable=im, cax=cax1, ticks=ticks, orientation="vertical")
        cbar1.ax.set_yticklabels(cl)

        return fig


def plot_experiment(label_image, experiment=2, image=None):
    """
    Generates plot of image and rgb mask with colorbar for specified experiment
    :param img_path: Path to input image
    :param mask_path: Path to input segmentation mask
    :param experiment: int Experimental setup (1,2 or 3)
    :return: plot of image and rgb mask with class colorbar
    """
    remapped_mask, classes_exp, colormap = remap_experiment(label_image, experiment)
    return plot_images(remapped_mask, colormap, classes_exp, image)


def to_comb_image(img, lbl, lbl_pred, experiment):
    with torch.no_grad():
        img, lbl, lbl_pred = to_numpy(img), to_numpy(lbl), to_numpy(lbl_pred)
        img = np.round(np.moveaxis(img, 0, -1) * 255).astype("uint8")
        lbl = mask_to_colormap(
            lbl,
            get_remapped_colormap(CLASS_INFO[experiment][0]),
            from_network=True,
            experiment=experiment,
        )
        lbl_pred = mask_to_colormap(
            lbl_pred,
            get_remapped_colormap(CLASS_INFO[experiment][0]),
            from_network=True,
            experiment=experiment,
        )
        comb_img = np.concatenate((img, lbl, lbl_pred), axis=1)
    return comb_img


def colourise_data(
    data: np.ndarray,  # NHW expected
    low: float = 0,
    high: float = 1,
    repeat: list = None,
    perf_colour: tuple = (255, 0, 0),
) -> np.ndarray:
    # perf_colour in RGB
    if high == -1:  # Scale by maximum present
        high = np.max(data)
    data = np.clip((data - low) / (high - low), 0, 1)
    colour_img = np.round(
        data[..., np.newaxis]
        * np.array(perf_colour)[np.newaxis, np.newaxis, np.newaxis, :]
    ).astype("uint8")
    if repeat is not None:
        colour_img = np.repeat(
            np.repeat(colour_img, repeat[0], axis=1), repeat[1], axis=2
        )
    return colour_img
