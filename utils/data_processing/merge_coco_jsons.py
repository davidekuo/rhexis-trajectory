import argparse
import json


def main():
    """
    Merges multiple COCO format JSON files into one COCO format
    json file by concatenating all the annotations in the input JSON files.
    We used this function to concatenate the annotations of the keypoints and
    bounding boxes of the COCO format JSON files of the training and validation
    sets.

    Args:
        -keypoint_path: path to the JSON file containing the keypoint annotations
        -bounding_box_path: path to the JSON file containing the bounding box annotations
    """
    parser = argparse.ArgumentParser(
        description="Merges keypoint and bounding box annotation JSONs."
    )
    parser.add_argument(
        "keypoint_path", type=str, help="path to keypoint annotation JSON"
    )
    parser.add_argument(
        "bbox_path", type=str, help="path to bounding box annotation JSON"
    )
    args = parser.parse_args()

    with open(args.keypoint_path, "r") as read_file:
        keypoints = json.load(read_file)

    with open(args.bbox_path, "r") as read_file:
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

    with open("merged_annotations.json", "w") as write_file:
        json.dump(bboxes, write_file)


if __name__ == "__main__":
    main()
