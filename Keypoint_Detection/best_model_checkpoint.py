import os
import glob
import json
import argparse

def best_model_checkpoint(output_dir: str) -> str:
  """
  This function inputs path to Detectron2 output directory containing
  metrics.json and model checkpoint .pth files.
  It reads metrics.json to identify the checkpoint with the highest
  keypoint AP and returns its filename

  Parameters:
    output_dir: string containing path to output directory
  
  Returns:
    max_kp_AP_checpoint: string of filename of best-performing checkpoint
  """
  
  with open(output_dir + "/metrics.json") as read_file:
    lines = read_file.readlines()
  
  checkpoint_list = glob.glob(os.path.join(output_dir,"*.pth"))

  max_kp_AP = 0
  max_kp_AP_checkpoint = ""

  for line in lines:
    json_dict = json.loads(line)
    if 'keypoints/AP' in json_dict:
      kp_AP = json_dict['keypoints/AP']
      iteration = json_dict['iteration']
      checkpoint = "model_" + f"{iteration}".zfill(7) + ".pth"
      res = [i for i in checkpoint_list if checkpoint in i]
      if kp_AP > max_kp_AP and len(res) == 1:
        max_kp_AP = kp_AP
        max_kp_AP_checkpoint = checkpoint

  return max_kp_AP_checkpoint

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("output_dir", type=str, help="path to Detectron2 output directory")
    args = parser.parse_args()

    print(best_model_checkpoint(args.output_dir))

if __name__ == "__main__":
    main()
