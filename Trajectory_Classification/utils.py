"""
Functions that generate trajectories
"""
import os
import sys
import glob
import pandas as pd


def normalize_coords(path_dfs, path_vid_sizes):
  pass

def load_all_pulls(DATA_LOC:str):
  output_folder = os.path.join(DATA_LOC,"OUTPUT")

  # Read in data
  files = os.listdir(output_folder)
  csvs = [csv for csv in files if csv.endswith(".csv")]
  path_dfs = [load_pull(os.path.join(output_folder,csv)) for csv in csvs]
  print(path_dfs)
  labels = [file_label(csv) for csv in csvs]

  # Read in pull_info.csv
  video_size_df = pd.read_csv(os.path.join(DATA_LOC,"pull_info.csv"))
  path_vid_sizes = [get_video_resolution(video_size_df, csv.split('_fea')[0]) for csv in csvs]

  # Normalize all coords to size of image
  for path in path_df:
    normalize_coords(path, path_vid_sizes)

  names = [csv.split('_fea')[0] for csv in csvs]
  return names, path_dfs, labels, path_vid_sizes



def load_pull(filename):
  '''Reads the coordinates from a pull file into a dataframe.

  Frames where a forceps was not detected are filtered out.

  Args:
    filename: The pull csv filepath to load from
  
  Returns:
    A pandas dataframe of the coordinates where the forceps is present, sorted
      by frame (return Y, X)
  '''
  data = pd.read_csv(filename)
  return data[data.key_L_x.notnull()].sort_values(by=['frame_num'])

def get_video_resolution(video_resolution_df, filename):
  col = video_resolution_df[filename]
  return (col[2], col[1])


def file_label(filename):
  '''Determines the label for data from a file based on the filename.

  Args:
    filename: The path filepath to determine based on

  Returns:
    A number representing the true class of the data:
      0 = Junior Resident
      1 = Senior Resident
      2 = Expert
  '''
  if filename.startswith("Medi"):
    return 0
  elif filename.startswith(("KY", "AC")):
    return 1
  elif filename.startswith(("Cataract", "SQ")):
    return 2
  else:
    raise Exception("Unhandled filetype: " + filename)