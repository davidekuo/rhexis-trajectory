import argparse
import os
import fnmatch
import random
import shutil

random.seed(0)


def main():
    """
    Generates training data by selecting X random jpgs from Y 
    random subdirectories of current directory and copies them to output 
    directory. It expects that all subdirectories of current directory contain 
    jpgs corresponding to frames of a cataract surgery video.

    Args:
        -nvideos: number of random directories (i.e. videos) to select.
         default -1 includes all directories/videos
        -nframes: number of random frames to select from each video.
         default is 20
        -outdir: output directory to copy selected frames to.
         default is ./training_data
    """
    parser = argparse.ArgumentParser(
        description="Generates training data by selecting X random jpgs from Y random subdirectories of current directory and copies them to output directory. Expects that all subdirectories of current directory contain jpgs corresponding to frames of a cataract surgery video."
    )
    parser.add_argument(
        "-nvideos",
        type=int,
        help="number of random directories (i.e. videos) to select: default -1 includes all directories/videos",
        default=-1,
    )
    parser.add_argument(
        "-nframes",
        type=int,
        help="number of random frames to select from each video: default is 20",
        default=20,
    )
    parser.add_argument(
        "-output",
        type=str,
        help="output directory: default is ./training_data",
        default="training_data",
    )
    args = parser.parse_args()

    # terminal output for debugging
    print("\nParameters:")
    print(
        f"args.nvideos = {args.nvideos} \nargs.nframes = {args.nframes} \nargs.output = {args.output}\n"
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
    if args.nvideos != -1:
        training_videos = random.sample(list(video_frames.keys()), args.nvideos)
    else:  # include all videos in directory
        training_videos = list(video_frames.keys())

    training_frames = []

    for video in training_videos:
        frames = random.sample(video_frames[video], args.nframes)
        training_frames += [os.path.join(video, frame) for frame in frames]

    # terminal output for debugging
    print(
        f"Generated training data: {len(training_videos)} videos, {len(set(training_frames))} unique frames."
    )
    print(f"\nVideos: {training_videos} \n\nFrames: {training_frames}\n")

    # create output directory for training data
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # copy training frames to training data output directory
    for frame in training_frames:
        shutil.copy(frame, args.output)


if __name__ == "__main__":
    main()
