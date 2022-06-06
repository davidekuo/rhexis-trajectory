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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin

DATA_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls"
REPO_LOC = "/content/drive/MyDrive/Projects/rhexis-trajectory"


def normalize_coords(path_df, path_vid_size):
    height, width = path_vid_size
    for column in path_df:
        if column.endswith(("_x", "_width")):
            path_df[column] = path_df[column] / width
        if column.endswith(("_y", "_height")):
            path_df[column] = path_df[column] / height


def drop_rows(path_df):
    # Keeps only every third row
    return path_df.iloc[::3, :]


def apply_moving_average(path_df):
    # Applies moving average with previoius 5 points
    for column in path_df:
        if column.endswith(("_x")) or column.endswith(("_y")):
            path_df[column] = path_df[column].rolling(5).mean()


def load_all_pulls(DATA_LOC: str):
    output_folder = os.path.join(DATA_LOC, "OUTPUT")

    # Read in data
    files = os.listdir(output_folder)
    csvs = [csv for csv in files if csv.endswith(".csv")]
    path_dfs = [load_pull(os.path.join(output_folder, csv)) for csv in csvs]
    # print(path_dfs)
    labels = [file_label(csv) for csv in csvs]

    # Read in pull_info.csv
    video_size_df = pd.read_csv(os.path.join(DATA_LOC, "pull_info.csv"))
    path_vid_sizes = [
        get_video_resolution(video_size_df, csv.split("_fea")[0]) for csv in csvs
    ]

    # Normalize all coords to size of image
    for path, path_vid_size in zip(path_dfs, path_vid_sizes):
        # path_dfs[i] = drop_rows(path)
        # path_dfs[i] = apply_moving_average(path)
        normalize_coords(path, path_vid_size)

    names = [csv.split("_fea")[0] for csv in csvs]
    return names, path_dfs, labels, path_vid_sizes


def load_pull(filename, header=None):
    """Reads the coordinates from a pull file into a dataframe.

    Frames where a forceps was not detected are filtered out.

    Args:
      filename: The pull csv filepath to load from

    Returns:
      A pandas dataframe of the coordinates where the forceps is present, sorted
        by frame (return Y, X)
    """
    data = pd.read_csv(filename, header=header) if header else pd.read_csv(filename)
    return data[data.key_L_x.notnull()].sort_values(by=["frame_num"])


def get_video_resolution(video_resolution_df, filename):
    col = video_resolution_df[filename]
    return (col[2], col[1])


def file_label(filename):
    """Determines the label for data from a file based on the filename.

    Args:
      filename: The path filepath to determine based on

    Returns:
      A number representing the true class of the data:
        0 = Resident
        1 = Expert
    """
    if filename.startswith(("Medi", "KY", "AC")):
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

    features = np.append(
        hist, np.array([length, mean_velocity, mean_accel, *pupil_stds])
    )

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

    for i in range(len(x) - 1):
        x1, x2 = x[i : i + 2]
        y1, y2 = y[i : i + 2]
        velocities.append(np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)))

    for i in range(len(velocities) - 1):
        accelerations.append(np.abs(velocities[i + 1] - velocities[i]))

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
    for i in range(len(x) - 2):
        x1, x2, x3 = x[i : i + 3]
        y1, y2, y3 = y[i : i + 3]
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
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
    bins = [i for i in range(0, 181, 180 // num_bins)]
    return np.histogram(angles, bins)


def stratified_split_data(X, y):
    np.random.seed(13)
    sss = StratifiedShuffleSplit(1, test_size=0.2)
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


def get_data_for_fixed_bins(n_bins):
    names, path_dfs, labels, sizes = load_all_pulls(DATA_LOC)
    X, y = np.stack(
        [featurize_pull(pull, n_bins) for pull in path_dfs], axis=0
    ), np.array(labels)
    return stratified_split_data(X, y)


def grid_search(clf, param_grid, X_train, y_train):
    search = GridSearchCV(make_custom_pipeline(clf, include_pca=True), param_grid, cv=5)
    search.fit(X_train, y_train)
    return search


def grid_search_with_bins(clf, param_grid, bin_range):
    names, path_dfs, labels, sizes = load_all_pulls(DATA_LOC)
    results = {}
    for num_bins in bin_range:
        X, y = np.stack(
            [featurize_pull(pull, num_bins) for pull in path_dfs], axis=0
        ), np.array(labels)
        X_train, X_test, y_train, y_test = stratified_split_data(X, y)
        search = GridSearchCV(
            make_custom_pipeline(clf, include_pca=True), param_grid, cv=5
        )
        search.fit(X_train, y_train)
        results[num_bins] = search
    best_bin = get_best_gs_w_bins_nbins(results)
    print(best_bin)
    X, y = np.stack(
        [featurize_pull(pull, best_bin) for pull in path_dfs], axis=0
    ), np.array(labels)
    X_train, X_test, y_train, y_test = stratified_split_data(X, y)
    return {
        "results": results,
        "best_search": get_best_gs_w_bins_search(results),
        "best_score": get_best_gs_w_bins_score(results),
        "best_params": get_best_gs_w_bins_params(results),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


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


def convert_path_to_img_matrix(path_df, fill_val_w_frame=False):
    """
    DEPRECATED. No longer in use since we are projecting onto a standardized pupil
    matrix instead of getting the raw pixel path.

    Converts a list of path coordinates to an image matrix.
    In order to handle complexity with floats, the size of the matrix is 10x the
    number of pixels, and each point is rounded to the nearest one-tenth pixel.

    The matrix is also bounded such that the bottom-left pixel in the trajectory
    is placed at (0, 0), by shifting all points by the minimum coordinate.

    Additionally, there are two modes in which the matrix can be filled:
      1. Boolean path matrix: Places a "1" in each position the forceps was
           at any point in the trajectory
      2. Frame number matrix: Places the frame number in which the forceps was
           at a location. This may help with incorporate temporal quality.

    Args:
      path_df: the path coordinate dataframe to convert
      fill_val_w_frame: Whether to fill in the frame number as the coordinate
        value. Defaults to False.

    Returns;
      The 2D np array representing the trajectory
    """
    lf_coords = path_df[["key_L_x", "key_L_y", "frame_num"]]
    lf_coords["key_L_x"] = lf_coords["key_L_x"].subtract(lf_coords["key_L_x"].min())
    lf_coords["key_L_y"] = lf_coords["key_L_y"].subtract(lf_coords["key_L_y"].min())
    lf_coords = np.rint(lf_coords * 10).astype(int)

    maxes = lf_coords.max()
    arr = np.zeros((maxes[0], maxes[1]))
    lf_np = lf_coords.to_numpy()
    arr[lf_np[:, 0] - 1, lf_np[:, 1] - 1] = lf_np[:, 2] if fill_val_w_frame else 1
    return arr


def convert_path_to_std_pupil_matrix(
    path_df, size=None, fill_val_w_frame=False, verbose=False
):
    """Maps a path to a normalized matrix representing position relative to the
    center of a pupil.

    Args:
      path_df: The path coordinate dataframe to convert
      vid_res: The resolution of the video the path was taken from, which helps
        determine if any clipping of the pupil has occurred
      size: If set, maps the pupil to a specified size matrix. If unset, uses
        the pupil height and width for the matrix.
      fill_val_w_frame: Whether to fill in the frame number as the coordinate
        value. Defaults to False.
    """
    # Determine the pupil size initial matrix
    pupil_width = abs(path_df["pupil_right_x"] - path_df["pupil_left_x"]).mean()
    pupil_height = abs(path_df["pupil_up_y"] - path_df["pupil_down_y"]).mean()
    arr = np.zeros(size or (pupil_width, pupil_height))
    arr_center_x, arr_center_y = (np.array(arr.shape) / 2).astype(int)

    for i in range(len(path_df)):
        # Calculate how much to translate forceps point to center pupil at center
        # of matrix
        translation_factor_x = path_df["pupil_left_x"].iloc[i]
        translation_factor_y = path_df["pupil_up_y"].iloc[i]
        if verbose:
            print("=======================================================")
            print("Translate: " + str((translation_factor_x, translation_factor_y)))

        # Calculate scale factor for forceps
        # Assume elliptical if tilted. Find max between both sides to ensure one side
        # is not clipped
        pcx = path_df["pupil_center_x"].iloc[i]
        pcy = path_df["pupil_center_y"].iloc[i]
        plx = path_df["pupil_left_x"].iloc[i]
        prx = path_df["pupil_right_x"].iloc[i]
        puy = path_df["pupil_up_y"].iloc[i]
        pdy = path_df["pupil_down_y"].iloc[i]

        radius_x = max(pcx - plx, prx - pcx)
        radius_y = max(pcy - puy, pdy - pcy)
        scale_factor_x = arr_center_x / radius_x
        scale_factor_y = arr_center_y / radius_y
        if verbose:
            print("Scale: " + str((scale_factor_x, scale_factor_y)))

        coord_l_x = int(
            (path_df["key_L_x"].iloc[i] - translation_factor_x) * scale_factor_x
        )
        coord_l_y = int(
            (path_df["key_L_y"].iloc[i] - translation_factor_y) * scale_factor_y
        )

        if verbose:
            print("Coord: " + str((coord_l_y, coord_l_x)))
            print("=======================================================")

        if coord_l_x > 0 and coord_l_y > 0:
            arr[coord_l_y, coord_l_x] = (
                path_df["frame_num"].iloc[i] if fill_val_w_frame * 10 else 1
            )
    return arr


def get_pupil_std_data(matrix_size=None, fill_val_w_frame=False):
    """Loads data and performs conversion of paths to standardized pupil centered
    matrices.

    Args:
      matrix_size: The matrix size to project onto for the conversion function
      fill_val_w_frame: Whether to use the frame number as the value in the matrix.
        Defaults to 1.

    Returns:
      Tuple of matrix list and corresponding labels
    """
    matrix_size = matrix_size or (100, 100)
    DATA_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls/OUTPUT/"
    files = os.listdir(DATA_LOC)
    # VIDEO_SIZE_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls/pull_info.csv"
    # video_size_df = pd.read_csv(VIDEO_SIZE_LOC)
    path_dfs = [load_pull(f"{DATA_LOC}/{file}") for file in files]
    # path_vid_sizes = [get_video_resolution(video_size_df, file.split('_fea')[0]) for file in files]
    labels = [file_label(file) for file in files]
    # path_matrices = [convert_path_to_std_pupil_matrix(df, res, matrix_size, fill_val_w_frame) for df, res in zip(path_dfs, path_vid_sizes)]
    path_matrices = [
        convert_path_to_std_pupil_matrix(df, matrix_size, fill_val_w_frame)
        for df in path_dfs
    ]
    return np.stack(path_matrices, axis=0), np.array(labels)


def get_pupil_std_data_traj(matrix_size=None, fill_val_w_frame=False):
    """Loads data and performs conversion of paths to standardized pupil centered
    matrices.

    Args:
      matrix_size: The matrix size to project onto for the conversion function
      fill_val_w_frame: Whether to use the frame number as the value in the matrix.
        Defaults to 1.

    Returns:
      Tuple of matrix list and corresponding labels
    """
    matrix_size = matrix_size or (100, 100)
    DATA_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls/OUTPUT_TRAJECTORIES/"
    files = os.listdir(DATA_LOC)
    # VIDEO_SIZE_LOC = "/content/drive/MyDrive/Rhexis/datasets/test_pulls/pull_info.csv"
    # video_size_df = pd.read_csv(VIDEO_SIZE_LOC)
    path_dfs = [load_pull(f"{DATA_LOC}/{file}", header=2) for file in files]
    # path_vid_sizes = [get_video_resolution(video_size_df, file.split('_fea')[0]) for file in files]
    labels = [file_label(file) for file in files]
    path_matrices = [
        convert_path_to_std_pupil_matrix(df, matrix_size, fill_val_w_frame)
        for df in path_dfs
    ]
    return np.stack(path_matrices, axis=0), np.array(labels)


class ImageFlatteningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.reshape(X, (-1, 10000))


def make_img_pipeline(clf, n_components=10):
    return make_pipeline(
        ImageFlatteningTransformer(),
        StandardScaler(),
        PCA(n_components=n_components, svd_solver="randomized", whiten=True),
        clf,
    )


def make_img_kernel_pipeline(clf, n_components=10):
    return make_pipeline(
        ImageFlatteningTransformer(),
        StandardScaler(),
        KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=None,
            fit_inverse_transform=True,
            alpha=0.1,
        ),
        clf,
    )


def grid_search_img(clf, kernelized, param_grid, X_train, y_train):
    pipe = make_img_kernel_pipeline(clf) if kernelized else make_img_pipeline(clf)
    search = GridSearchCV(pipe, param_grid, cv=4)
    search.fit(X_train, y_train)
    return search


#####################################
### DEPRECATED METHODS            ###
#####################################


def pad_matrices(path_matrices):
    """
    DEPRECATED. No longer in use since we are projecting onto same-size matrices.

    Pads the right and up sides of a matrix with 0s to make all matrices in a
    list the same shape. Preserves the property that the bottom-left most point
    in the trajectory will be at (0,0)

    Args:
      path_matrices: List of matrices to pad

    Returns:
      A list of matrices of the same size
    """
    max_x = max(map(lambda x: x.shape[0], path_matrices))
    max_y = max(map(lambda x: x.shape[1], path_matrices))
    for i, m in enumerate(path_matrices):
        path_matrices[i] = np.pad(
            m, ((0, max_x - m.shape[0]), (0, max_y - m.shape[1])), mode="constant"
        )
        print(path_matrices[i].shape)
    return path_matrices


def featurize_trajectory(trajectory, label, transform_func=None):
    """
    DEPRECATED. No longer in use since we are using paths instead of trajectories.

    Splits a trajectory by pulls and featurizes each pull. Each pull is assigned
    the input label.

    Args:
      trajectory: The trajectory dataframe to split into featurized pulls
      label: The label to assign each pull for the trajectory
      transform_func: Custom transformation to use during featurization. Defaults
          to `featurize_pull`.

    Return:
      X: The feature rows
      y: The labels for each row
    """
    transform_func = transform_func or featurize_pull
    print(transform_func)
    X = [transform_func(df) for val, df in trajectory.groupby("pull")]
    y = np.repeat(label, len(X))
    return np.array(X), y


def featurize_trajectories(trajectories, transform_func=None):
    """
    DEPRECATED. No longer in use since we are using paths instead of trajectories.

    Converts a list of trajectories into a feature matrix and label vector

    Args:
      trajectory: The trajctories list to split into featurized pulls
      transform_func: Custom transformation to use during featurization. Defaults
          to `featurize_pull`.

    Return:
      X: The feature matrix
      y: The label vector
    """
    featurized_trajectory_list = [
        featurize_trajectory(t, i, transform_func) for i, t in enumerate(trajectories)
    ]
    X = np.row_stack([x for x, y in featurized_trajectory_list])
    y = np.concatenate([y for x, y in featurized_trajectory_list])
    return X, y
