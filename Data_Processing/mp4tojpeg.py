import os
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Extracts frames of filename.mp4 into output directory with JPEGs named filename_frame#.jpg. Default output directory is [input file directory]/[filename]"
    )
    parser.add_argument("input", type=str, help="path to input mp4 file")
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()

    input_list = args.input.split("/")
    filename = input_list[-1][0:-4]
    # args.input.split('/') converts string into list of strings split by '/'
    # filename.mp4 will be the last element of this list: index -1
    # omitting the last 4 characters of filename.mp4 (i.e. ".mp4") gives filename

    input_dir = "/".join(input_list[0:-1]) if len(input_list) > 1 else "."
    # if input actually contains a path (has at least 1 '/')
    # input directory will be all elements of input_list except the last
    # join list elements back into a string with elements separated by '/'
    # otherwise, input is a file in current directory: set input directory to '.'

    # create output directory - default is [input file directory]/[filename]
    output_dir = args.output if args.output else f"{input_dir}/{filename}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nParameters:")
    print(f"args.input = {args.input} | args.output = {args.output}")
    print(
        f"input directory: {input_dir} | output directory: {output_dir} | filename: {filename}"
    )
    print(f"output file format: {output_dir}/{filename}_frame#.jpg\n")

    vidcap = cv2.VideoCapture(args.input)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_dir}/{filename}_{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
    print(f"{count} frames extracted from {args.input} into {output_dir}!\n")


if __name__ == "__main__":
    main()
