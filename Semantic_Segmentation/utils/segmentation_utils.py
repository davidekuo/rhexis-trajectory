"""
segmentation_utils.py

This py file contains modified code from:
https://github.com/RViMLab/MICCAI2021_Cataract_semantic_segmentation
"""

# Semantic Segmentation Utils
import os
import json
import pathlib
from utils import *
import torch

def configure_segmentation_model(task: int):
  """
  This function configures the semantic segmentation model and returns a model
  object with the loaded weights.

  Parameters:
    task: int - The task we would like this model to perform as defined by the
      CADIS dataset. 
      https://www.sciencedirect.com/science/article/pii/S1361841521000992#tbl0005
      task = 1 indicates task 1
      task = 2 indicates task 2
      task = 3 indicates task 3

  Returns:
    model: Model object loaded to pretrained weights from the MICCAI2021
    Cataract Semantic Segmentation Model.
      https://github.com/RViMLab/MICCAI2021_Cataract_semantic_segmentation

  """
  # Assert input is in correct range
  assert (task >= 1) or (task <= 3)

  # Collect the path to the correct saved model checkpoint (specified by task)
  chkpt_loc = os.path.join(os.getcwd(), "segmentation_models",
    f"model_task{task}", "chkpts", "chkpt_best.pt")

  # Collect the path to the correct saved model config file
  config_loc = os.path.join(os.getcwd(), "segmentation_models",
    f"model_task{task}", f"OCRNet_pretrained_t{task}.json")

  # Create config object
  config = parse_config(config_loc,0)

  # Select model class as OCRNet
  model_class = globals()[config['graph']['model']]

  # Utilize this config object to create the model object
  model = model_class(config=config['graph'], experiment=task)

  # Turn off intermediate output
  model.get_intermediate = False

  print("Loading model from saved checkpoint...")
  
  # Load in model checkpoint
  checkpoint = torch.load(chkpt_loc, 'cuda:{}'.format(config['gpu_device']))
  
  # Set model to checkpoint
  model.load_state_dict(checkpoint['model_state_dict'], strict=False)

  print()
  print("Model successfully loaded from:")
  print(f"{chkpt_loc}")
  return model

def parse_config(file_path, device):
  """
  This function is a modified version of the function from the 
  MICCAI2021_Cataract_Semantic_Segmentation repo. Link to original version:
  https://github.com/RViMLab/MICCAI2021_Cataract_semantic_segmentation/blob/main/utils/utils.py
  """
  # Load config
  try:
      with open(file_path, 'r') as f:
          config_dict = json.load(f)
  except FileNotFoundError:
      print("Configuration file not found at given path '{}'".format(file_path))
      exit(1)

  # Fill in GPU device if applicable
  if device >= 0:  # Only update config if user entered a device (default otherwise -1)
      config_dict['gpu_device'] = device

  # Make sure all necessary default values exist
  default_dict = DEFAULT_CONFIG_DICT.copy()
  default_dict.update(config_dict)  # Keeps all default values not overwritten by the passed config
  nested_default_dicts = DEFAULT_CONFIG_NESTED_DICT.copy()
  for k, v in nested_default_dicts.items():  # Go through the nested dicts, set as default first, then update
      default_dict[k] = v  # reset to default values
      default_dict[k].update(config_dict[k])  # Overwrite defaults with the passed config values

  # Extra config bits needed
  default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment']

  return default_dict