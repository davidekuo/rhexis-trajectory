import glob
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser("Generate dense optical flow GIF from PNGs")
    parser.add_argument('input_dir', help="directory containing input PNGs")
    parser.add_argument('output', help="GIF output file path", default="optical_flow.gif")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    print("input dir PNGs: ", sorted(glob.glob(input_dir + "/*.png")))
    imgs = (Image.open(f) for f in sorted(glob.glob(input_dir + "/*.png")))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=output_dir, format='GIF', append_images=imgs,
            save_all=True, duration=200, loop=0)


if __name__ == '__main__':
    main()