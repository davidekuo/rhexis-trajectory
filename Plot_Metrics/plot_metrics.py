import json
import pandas as pd
import matplotlib.pyplot as plt
from plot_metrics_utils import json2dataframes, plot_kp_bbox_AP_loss

def main():
    """
    This program generates plots of keypoints/AP, bbox/AP, and total loss for all Detectron2 experiments
    """
    # Convert metrics.jsons for R50, R101, X101 with and without data augmentation
    # into Pandas Dataframes

    # R50
    R50_dataset1000 = 'R50_dataset1000_no_augment_iter5000_metrics.json'
    R50_dataset1000_AP, R50_dataset1000_loss = json2dataframes(R50_dataset1000)

    R50_dataset1000_augment = 'R50_dataset1000_augment_iter5000_metrics.json'
    R50_dataset1000_augment_AP, R50_dataset1000_augment_loss = json2dataframes(R50_dataset1000_augment)

    # R101
    R101_dataset1000 = 'R101_dataset1000_no_augment_iter5000_metrics.json'
    R101_dataset1000_AP, R101_dataset1000_loss = json2dataframes(R101_dataset1000)

    R101_dataset1000_augment = 'R101_dataset1000_augment_iter5000_metrics.json'
    R101_dataset1000_augment_AP, R101_dataset1000_augment_loss = json2dataframes(R101_dataset1000_augment)

    # X101
    X101_dataset1000 = 'X101_dataset1000_no_augment_iter5000_metrics.json'
    X101_dataset1000_AP, X101_dataset1000_loss = json2dataframes(X101_dataset1000)

    X101_dataset1000_augment = 'X101_dataset1000_augment_iter5000_metrics.json'
    X101_dataset1000_augment_AP, X101_dataset1000_augment_loss = json2dataframes(X101_dataset1000_augment)

    # Generate plots
    all_data = [(R50_dataset1000_AP, R50_dataset1000_loss, 'R50 without augmentation'), 
            (R50_dataset1000_augment_AP, R50_dataset1000_augment_loss, 'R50 with augmentation'),
            (R101_dataset1000_AP, R101_dataset1000_loss, 'R101 without augmentation'),
            (R101_dataset1000_augment_AP, R101_dataset1000_augment_loss, 'R101 with augmentation'),
            (X101_dataset1000_AP, X101_dataset1000_loss, 'X101 without augmentation'),
            (X101_dataset1000_augment_AP, X101_dataset1000_augment_loss, 'X101 with augmentation')]

    plot_kp_bbox_AP_loss(all_data)


if __name__ == "__main__":
    main()
