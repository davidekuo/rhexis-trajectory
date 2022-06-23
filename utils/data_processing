import os
import fnmatch
import random
import shutil
import json

def convert_line_to_two_points(line: str):
    """
    A function that reads in the line containing the point values and alters the line to contain
    two points instead of one if necessary.

    Parameters:
        line: String containing the line of the xml file that contains point values

    Returns:
        String with two instances of points
        - If original string had two instances of point values, the string will not be changed
        - If original string had one instance of point values, the string will be altered
    """

    # Find where the point variable starts
    point_index = line.find('points="')

    # Save the rest of the string in a split
    remaining_str = line[point_index:].split(" ")

    # Find the start and end index of the point values
    point_value_start = remaining_str[0].find('"') + 1
    point_value_end = remaining_str[0].find('"', point_value_start + 1)

    # Collect the point value containing string
    point_values = remaining_str[0][point_value_start:point_value_end]

    # If string does not contain the ; seperator then add the duplicate point values
    if ";" not in point_values:
        new_line = line[0:point_index]
        new_line = new_line + 'points="'
        new_line = new_line + point_values
        new_line = new_line + ";"
        new_line = new_line + point_values + '"'
        new_line = new_line + " " + remaining_str[1]
        line = new_line

    return line

def convert_xml_one_point_to_two_points(xml_path:str, outdir:str = None):
    """
    Converts an XML file that may contain only one keypoint in some frames to
    an XML file that contains two keypoints in each frame. The new XML file is
    named TWO_POINTS_<original_xml_filename>.xml and is saved in the directory
    specified by outdir.

    Parameters:
        -xml_path: path to the XML file to convert
        -outdir: path to the output directory to save the new XML file to
            default is filename of .xml file placed within the current directory
    """

    # Get the xml filename
    xml_filename = xml_path.split("/")[-1]

    newtext = ""
    with open(xml_path) as file:
        # Read in lines from the file
        lines = file.readlines()

        # For each line
        for line in lines:
            # if the line contains the "point=", run the line through the function
            if "points=" in line:
                line = convert_line_to_two_points(line)

            # Add the line to the new text
            newtext = newtext + line

    # Determine if the outdir directory exists, if not create it

    output_filename = ""
    if outdir is not "":
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        output_filename = os.path.join(outdir, "TWO_POINTS_" + xml_filename)
    else:
        output_filename = os.path.join("TWO_POINTS_" + xml_filename)
    
    # Now write the newtext variable to an output file
    with open(output_filename) as file:
        file.write(newtext)


def mp4_to_jpg(mp4_path:str, outdir:str = ""):
    """
    Converts an MP4 file to a series of JPEG images. The new images
    are named utilizing the original name of the MP4 file and their
    corresponding frame number.

    Parameters:
        -mp4_path: path to the MP4 file to convert to JPEGs
        -outdir: path to the output directory to save the JPEGs to
            default is filename of .mp4 file placed within the current directory
    """

    # .split('/') converts string into list of strings split by '/'
    input_list = mp4_path.split("/")
    # filename.mp4 will be the last element of this list: index -1
    # omitting the last 4 characters of filename.mp4 (i.e. ".mp4") gives filename
    filename = input_list[-1][0:-4]
    
    
    input_dir = "/".join(input_list[0:-1]) if len(input_list) > 1 else "."
    # if input actually contains a path (has at least 1 '/')
    # input directory will be all elements of input_list except the last
    # join list elements back into a string with elements separated by '/'
    # otherwise, input is a file in current directory: set input directory to '.'

    # create output directory - default is [input file directory]/[filename]
    output_dir = outdir if outdir=="" else f"{input_dir}/{filename}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nParameters:")
    print(f"mp4_path = {mp4_path} | outdir = {output_dir}")
    print(
        f"input directory: {input_dir} | output directory: {output_dir} | filename: {filename}"
    )
    print(f"output file format: {output_dir}/{filename}_frame#.jpg\n")

    vidcap = cv2.VideoCapture(mp4_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_dir}/{filename}_{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
    print(f"{count} frames extracted from {mp4_path} into {output_dir}!\n")

def merge_coco_json(keypoints_json_path:str, bbox_json_path:str, output_json_path:str = "./coco_merged.json"):
    """
    Merges multiple COCO format JSON files into one COCO format
    json file by concatenating all the annotations in the input JSON files.
    We used this function to concatenate the annotations of the keypoints and
    bounding boxes of the COCO format JSON files of the training and validation
    sets.

    Parameters:
        -keypoints_json_path: path to the JSON file containing the keypoint annotations

        -bounding_box_path: path to the JSON file containing the bounding box annotations

        -output_json_path: path to the output JSON file
         default is "./coco_merged.json"
    """

    with open(keypoints_json_path, "r") as read_file:
        keypoints = json.load(read_file)

    with open(bounding_box_path, "r") as read_file:
        bboxes = json.load(read_file)

    imageid2keypoints = {
        annot["image_id"]: annot["keypoints"] for annot in keypoints["annotations"]
    }

    for annot in bboxes["annotations"]:
        annot["category_id"] = 1
        imageid = annot["image_id"]
        annot["num_keypoints"] = 2
        if imageid in imageid2keypoints:
            annot["keypoints"] = imageid2keypoints[imageid]
        else:
            annot["keypoints"] = [0, 0, 0, 0, 0, 0]

    bboxes["categories"] = keypoints["categories"]

    with open(output_json_path, "w") as write_file:
        json.dump(bboxes, write_file)


def generate_training_data(nvideos: int = -1, nframes:int = 20, outdir:str = "./training_data"):
    """
    Generates training data by selecting X random jpgs from Y 
    random subdirectories of current directory and copies them to output 
    directory. It expects that all subdirectories of current directory contain 
    jpgs corresponding to frames of a cataract surgery video.

    Parameters:
        -nvideos: number of random directories (i.e. videos) to select.
         default -1 includes all directories/videos

        -nframes: number of random frames to select from each video.
         default is 20

        -outdir: output directory to copy selected frames to.
         default is ./training_data
    """

    # terminal output for debugging
    print("\nParameters:")
    print(
        f"nvideos = {nvideos} \nnframes = {nframes} \noutdir = {outdir}\n"
    )

    # dictionary storing all subdirectories (i.e. videos) in current directory and names of all jpgs (i.e. frames)
    # in for each subdirectory/video
    video_frames = {}

    # os.walk(directory) returns a generator
    # next(os.walk(directory)) returns 3-tuple of
    # - path to current directory,
    # - list of all subdirectories,
    # - list of all files in current directory
    # fnmatch generates a list of all .jpg files in each subdirectory
    # which is stored in the video_frames dictionary
    path, dirs, files = next(os.walk("."))
    for dir in dirs:
        subdir = os.path.join(path, dir)
        subjpgs = fnmatch.filter(os.listdir(subdir), "*.jpg")
        video_frames[subdir] = subjpgs

    # random.sample(somelist, someint) generates a list of someint elements of somelist
    # randomly sampled w/o replacement
    if nvideos != -1:
        training_videos = random.sample(list(video_frames.keys()), nvideos)
    else:  # include all videos in directory
        training_videos = list(video_frames.keys())

    training_frames = []

    for video in training_videos:
        frames = random.sample(video_frames[video], nframes)
        training_frames += [os.path.join(video, frame) for frame in frames]

    # terminal output for debugging
    print(
        f"Generated training data: {len(training_videos)} videos, {len(set(training_frames))} unique frames."
    )
    print(f"\nVideos: {training_videos} \n\nFrames: {training_frames}\n")

    # create output directory for training data
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # copy training frames to training data output directory
    for frame in training_frames:
        shutil.copy(frame, outdir)