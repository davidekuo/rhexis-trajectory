import json
import pandas as pd
import matplotlib.pyplot as plt

def json2dataframes(file_path: str):
    """
    This function takes in the path to metrics.json from Detectron2 training
    and returns 2 Pandas dataframes of validation AP and loss data respectively

    Parameters:
        file_path: string containing path to metrics.json file of interest

    Returns:
        average_precision: Pandas dataframe containing validation AP data
        losses: Pandas dataframe containing validation loss data

    """
    with open(file_path) as read_file:
        lines = read_file.readlines()

    AP, L = [], []
    for line in lines:
        json_dict = json.loads(line)
        if 'bbox/AP' in json_dict:
            AP.append(json_dict)
        else:
            L.append(json_dict)

    average_precision = pd.DataFrame(AP)
    losses = pd.DataFrame(L)

    return average_precision, losses


def plot_kp_bbox_AP_loss(ap_loss_labels):
    """
    This function takes a list of tuples:
      (DataFrame average_precision, DataFrame loss, label string)
    and generates Matplotlib plots of keypoints/AP and bbox/AP

    Parameters:
        data_labels: List of tuples (DataFrame average_precision, DataFrame loss, label string)
        for generating Matplotlib plots

    Returns:
        N/A
    """

    # Plot keypoints/AP
    for AP, loss, label in ap_loss_labels:
        plt.plot(AP['iteration'], AP['keypoints/AP'], label=f"{label}")
    plt.title("Validation Keypoints/Average Precision")
    plt.xlabel("Iterations")
    plt.ylabel("AP")
    plt.legend()
    # plt.show()
    plt.savefig('kp_AP.png', bbox_inches='tight')
    plt.clf()

    # Plot bbox/AP
    for AP, loss, label in ap_loss_labels:
        plt.plot(AP['iteration'], AP['bbox/AP'], label=f"{label}")
    plt.title("Validation Bbox/Average Precision")
    plt.xlabel("Iterations")
    plt.ylabel("AP")
    plt.legend()
    # plt.show()
    plt.savefig('bbox_AP.png', bbox_inches='tight')
    plt.clf()
#
    # Plot total loss
    for AP, loss, label in ap_loss_labels:
        plt.plot(loss['iteration'], loss['total_loss'], label=f"{label}")
    plt.title("Total Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('loss.png', bbox_inches='tight')
    plt.clf()
