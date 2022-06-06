import os
import glob as gb
import argparse


def main():
    parser = argparse.ArgumentParser("Run RAFT inference")
    parser.add_argument("path", help="path to dataset")
    args = parser.parse_args()

    data_path = args.path  # "/path/to/dataset"
    gap = [1, 2]
    reverse = [0, 1]
    rgbpath = data_path + "/JPEGImages"  # path to the dataset
    folder = gb.glob(os.path.join(rgbpath, "*"))

    for r in reverse:
        for g in gap:
            for f in folder:
                print("===> Running {}, gap {}".format(f, g))
                model = "raft-things.pth"  # model
                if r == 1:
                    raw_outroot = data_path + "/Flows_gap-{}/".format(
                        g
                    )  # where to raw flow
                    outroot = data_path + "/FlowImages_gap-{}/".format(
                        g
                    )  # where to save the image flow
                elif r == 0:
                    raw_outroot = data_path + "/Flows_gap{}/".format(
                        g
                    )  # where to raw flow
                    outroot = data_path + "/FlowImages_gap{}/".format(
                        g
                    )  # where to save the image flow
                os.system(
                    "python predict.py "
                    "--gap {} --model {} --path {} "
                    "--outroot {} --reverse {} --raw_outroot {}".format(
                        g, model, f, outroot, r, raw_outroot
                    )
                )


if __name__ == "__main__":
    main()
