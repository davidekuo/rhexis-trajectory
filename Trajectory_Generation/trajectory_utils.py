"""
Functions that generate trajectories
"""
# TODO: Figure out where to place this function and delete file

import os
import sys
import glob


def get_folders_from_substrings(DATA_LOC, substring):

    # Collect all folders in the DATA_LOC
    folder_list = glob.glob(os.path.join(DATA_LOC, "*/"))

    # Collect all folders that contain the substring
    subdir_list = [
        str.split(name, os.sep)[-2] for name in folder_list if substring in name
    ]

    # return the list
    return subdir_list
