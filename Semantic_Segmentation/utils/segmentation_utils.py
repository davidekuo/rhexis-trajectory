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
import numpy as np
import glob
import cv2
import time

def create_label_images(labels, img_data, DATA_LOC):
  
  # For each image
  for i in range(labels.shape[0]):

    # Determine the name of this label
    


def create_labels(model, x):
  """
  Reads in our data and creates labels by running them through a forward pass
  """
  # Intialize return structure
  labels = []

  # Syncronize cuda
  #torch.cuda.synchronize()

  # Create batches
  num_batches = 50
  batches = np.split(x, num_batches)
  print("Creating Labels:")


  batches = batches[0:2]

  for i, x_batch in enumerate(batches):
    start_time = time.time()
    with torch.no_grad():
      print(f"Batch {i+1} of {len(batches)}")

      print("-Applying forward pass")
      logits = model.forward(torch.tensor(x_batch,device=torch.device('cuda:0')).float())

      print("-Calculating softmax and outputing labels")
      # Compute the label predictions using a 2D softmax
      label_predictions = torch.argmax(torch.nn.Softmax2d()(logits), dim=1).cpu().numpy()

      # Append results
      labels.append(label_predictions)
    print(f"-Completed after {time.time() - start_time} seconds")

  # Concatenate all batch results and return
  return np.concatenate(labels,axis = 0)







def read_in_images(DATA_LOC: str):
  """
  Reads in images from the dataset location and returns a numpy array containing
  the images of shape (NxCxHxW), where N is the number of images, C is the
  number of channels, and H + W are sizes of spatial dimensions.
  """

  # Glob the data together
  X = []

  # Collect all *.jpg files
  train_files = glob.glob(os.path.join(DATA_LOC, "train_set","images","*.jpg"))
  val_files = glob.glob(os.path.join(DATA_LOC, "val_set","images","*.jpg"))
  test_files = glob.glob(os.path.join(DATA_LOC, "test_set","images","*.jpg"))

  # Concatentate the list of files
  file_list = train_files + val_files + test_files

  # save the sizes of each file so we can resize at the end
  size_list = []
  print("Loading in images:")
  print(f"{len(file_list)} jpg files detected")
  for f in file_list:
    image= cv2.imread(f)
    size_list.append((image.shape[0],image.shape[1]))
    X.append(np.array(cv2.resize(image, (960, 540))))


  # Stack images into 4D NxHxWxC
  stack = np.array(X)

  # Rearange channel dimension before returning
  # NxHxWxC --> NxCxWxH --> NxCxHxW
  stack = np.swapaxes(stack,1,3).swapaxes(2,3)
  print()
  print("Images loaded successfully")
  return stack, (file_list, size_list)



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

  # Update config dict so that weights are loaded in on cuda
  config['gpu_device'] = 0

  # Utilize this config object to create the model object
  model = model_class(config=config['graph'], experiment=task)

  # Turn off intermediate output
  model.get_intermediate = False

  print("Loading model from saved checkpoint...")
  
  # Load in model checkpoint
  # this is required if it checkpoint trained on one device and now is loaded on a different device
  # https://github.com/pytorch/pytorch/issues/15541
  map_location = 'cuda:{}'.format(config['gpu_device'])
  checkpoint = torch.load(str(chkpt_loc), map_location)
  model.load_state_dict(checkpoint['model_state_dict'], strict=False) # todo fix this

  print("Model successfully loaded from:")
  print(f"{chkpt_loc}")

  # GPU Disabled for now: Transfer between GPU and CPU for results takes way to
  # long.
  # Move model to the GPU
  model.cuda()

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