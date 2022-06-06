"""
Functions that generate trajectories
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls"
REPO_LOC = "/content/drive/MyDrive/Projects/rhexis-trajectory"

def normalize_coords(path_df, path_vid_size):
  height, width = path_vid_size
  for column in path_df:
    if column.endswith(("_x", "_width")):
      path_df[column] = path_df[column] / width
    if column.endswith(("_y", "_height")):
      path_df[column] = path_df[column] / height

def load_all_pulls(DATA_LOC:str):
  output_folder = os.path.join(DATA_LOC,"OUTPUT")

  # Read in data
  files = os.listdir(output_folder)
  csvs = [csv for csv in files if csv.endswith(".csv")]
  path_dfs = [load_pull(os.path.join(output_folder,csv)) for csv in csvs]
  # print(path_dfs)
  labels = [file_label(csv) for csv in csvs]

  # Read in pull_info.csv
  video_size_df = pd.read_csv(os.path.join(DATA_LOC,"pull_info.csv"))
  path_vid_sizes = [get_video_resolution(video_size_df, csv.split('_fea')[0]) for csv in csvs]

  # Normalize all coords to size of image
  for path, path_vid_size in zip(path_dfs, path_vid_sizes):
    normalize_coords(path, path_vid_size)

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
  # if filename.startswith("Medi"):
  #   return 0
  # elif filename.startswith(("KY", "AC")):
  #   return 1
  # elif filename.startswith(("Cataract", "SQ")):
  #   return 2
  if filename.startswith(("Medi","KY", "AC")):
    return 0
  elif filename.startswith(("Cataract", "SQ")):
    return 1
  else:
    raise Exception("Unhandled filetype: " + filename)

def featurize_pull(pull, num_bins=20, normalize=False):
  """
  Performs all featurizations for an individual pull

  Args:
    pull: The pull dataframe to convert to a feature row

  Returns:
    row: A feature row for the pull
  """
  angles = pull_to_angles(pull)
  hist, bins = angles_to_bins(angles, num_bins)

  length = pull_to_length(pull)
  mean_velocity, mean_accel = pull_to_velocities_and_accelerations(pull)
  pupil_stds = pull_to_pupil_stddev(pull)

  features = np.append(hist, np.array([length, mean_velocity, mean_accel, *pupil_stds]))

  return features

def pull_to_pupil_stddev(pull):
  return np.std(pull["pupil_center_x"]), np.std(pull["pupil_center_y"])

def pull_to_length(pull):
  """
  Args:
    pull: The dataframe for the pull to conver to velocities

  Return:
    velocities: velocities calculated as distance traveled per frame
  """
  return len(pull)

def pull_to_velocities_and_accelerations(pull):
  """
  Args:
    pull: The dataframe for the pull to conver to velocities
  
  Return:
    mean velocity: velocity calculated as distance traveled per frame
    mean acceleration: acceleration calculated as change in velocity between frames
  """
  x, y = pull.key_L_x, pull.key_L_y
  velocities = []
  accelerations = []

  for i in range(len(x)-1):
    x1, x2 = x[i:i+2]
    y1, y2 = y[i:i+2]
    velocities.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))

  for i in range(len(velocities) - 1):
    accelerations.append(np.abs(velocities[i+1] - velocities[i]))

  return (np.mean(velocities), np.mean(accelerations))

def pull_to_angles(pull):
  """
  Args:
    pull: The dataframe for the pull to convert to angles

  Return:
    angles: The angles for each triple of datapoints
  """
  x, y = pull.key_L_x, pull.key_L_y
  angles = []
  for i in range(len(x)-2):
    x1, x2, x3 = x[i:i+3]
    y1, y2, y3 = y[i:i+3]
    v1 = np.array([x1-x2, y1-y2])
    v2 = np.array([x3-x2, y3-y2])
    angle_nocos = np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))
    angle_floor = np.where(angle_nocos < -1, -1.0, angle_nocos)
    angle_ceil = np.where(angle_floor > 1, 1.0, angle_floor) 
    angle = np.arccos(angle_ceil) * 180 / np.pi
    angles.append(angle)
  return angles


def angles_to_bins(angles, num_bins=20):
  """
  Args:
    angles: the list of angles to bin

  Returns:
    histogram: histogram values
    bins: histogram bins
  """
  bins = [i for i in range(0, 181, 180//num_bins)] 
  return np.histogram(angles, bins)

def stratified_split_data(X, y):
  np.random.seed(13)
  sss = StratifiedShuffleSplit(1, test_size=.2)
  train_ind, test_ind = next(sss.split(X, y))
  X_train, X_test = X[train_ind], X[test_ind]
  y_train, y_test = y[train_ind], y[test_ind]
  return X_train, X_test, y_train, y_test

def make_custom_pipeline(clf, include_pca=False, num_components=0):
  scaler = StandardScaler()
  pca = PCA()
  if include_pca:
    return make_pipeline(scaler, pca, clf)
  else:
    return make_pipeline(scaler, clf)

def run_pipeline_test():
  X, y = np.stack([featurize_pull(pull, num_bins) for pull in path_dfs], axis=0), np.array(labels)
  X_train, X_test, y_train, y_test = stratified_split_data(X, y)
  search = GridSearchCV(make_custom_pipeline(clf, include_pca=True), param_grid, cv=5)
  search.fit(X_train, y_train)

def grid_search_with_bins(clf, param_grid, bin_range):
  names, path_dfs, labels, sizes = load_all_pulls(DATA_LOC)
  results = {}
  for num_bins in bin_range:
    X, y = np.stack([featurize_pull(pull, num_bins) for pull in path_dfs], axis=0), np.array(labels)
    X_train, X_test, y_train, y_test = stratified_split_data(X, y)
    search = GridSearchCV(make_custom_pipeline(clf, include_pca=True), param_grid, cv=5)
    search.fit(X_train, y_train)
    results[num_bins] = search
  best_bin = get_best_gs_w_bins_nbins(results)
  print(best_bin)
  X, y = np.stack([featurize_pull(pull, best_bin) for pull in path_dfs], axis=0), np.array(labels)
  X_train, X_test, y_train, y_test = stratified_split_data(X, y)
  return {"results": results,
          "best_search": get_best_gs_w_bins_search(results),
          "best_score": get_best_gs_w_bins_score(results),
          "best_params": get_best_gs_w_bins_params(results),
          "X_train": X_train,
          "X_test": X_test,
          "y_train": y_train,
          "y_test": y_test}

def get_best_gs_w_bins_search(results):
  return max(results.values(), key=lambda x: x.best_score_)

def get_best_gs_w_bins_score(results):
  return max(map(lambda x: x.best_score_, results.values()))

def get_best_gs_w_bins_params(results):
  pair = max(results.items(), key=lambda k_v: k_v[1].best_score_)
  best_dict = pair[1].best_params_
  best_dict["data__n_bins"] = pair[0]
  return best_dict

def get_best_gs_w_bins_nbins(results):
  return max(results.items(), key=lambda k_v: k_v[1].best_score_)[0]







