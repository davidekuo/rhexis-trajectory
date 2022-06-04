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
import math

def semantic_segmentation(DATA_LOC, subdir_names, task = 2, test_mode=True, use_image_subdir = True):
  # Set up model
  model = configure_segmentation_model(task)

  # Read in images
  x, img_data = read_in_images(DATA_LOC, subdir_names, use_image_subdir)

  # Create labels
  create_labels(model, x, img_data, DATA_LOC, subdir_names, test_mode, use_image_subdir)



def make_label_directories(DATA_LOC, subdir_names = []):

  if not subdir_names:
    subdir_names = ['train_set', 'val_set', 'test_set']
  
  # Create these folders
  for name in subdir_names:
    folder_to_create = os.path.join(DATA_LOC, name, "labels")
    
    try:
      os.mkdir(folder_to_create)
    except OSError as error:
      pass
      #print(f"{name} label folder already exists | overwriting labels")

def create_label_images(labels, img_data, DATA_LOC, subdir_names = [], use_image_subdir = True):
  """
  Creates images of all of our labels
  """
  # Keep track of how many images saved successfully and which ones did not
  success_count = 0

  # Keeps track of which images
  failures = []

  # Unpack img_data
  file_list, size_list = img_data

  # Make the label directories, if they do not exist
  make_label_directories(DATA_LOC, subdir_names)

  # For each image
  for i in range(labels.shape[0]):
    # Get the current file name
    f = file_list[i]

    # Determine the name of this image
    file_name = str.split(str.split(f, os.sep)[-1],".")[0]
    
    # Determine which set the image is in
    image_set = ""
    if use_image_subdir:
      image_set = str.split(f, os.sep)[-3]
    else:
      image_set = str.split(f, os.sep)[-2]

    # Create final image path
    label_path = os.path.join(DATA_LOC, image_set, "labels", file_name + ".png")

    # Resize image to apporiate dimensions (uses interpolation method 
    # nearest neighbors so label values are unchanged)
    # NOTE: for some reason the label sizes got flipped??, reversed it
    label_resized = cv2.resize(labels[i], (size_list[i][1],size_list[i][0]), 
      interpolation = cv2.INTER_NEAREST)

    # Write image to path location
    if(cv2.imwrite(label_path, label_resized)):
      success_count += 1
    else:
      failures.append(f)

  if len(failures) > 0:
    print("Images that failed to generate labels:")
    for fail in failures:
      print(fail)
  
  print("")
  print(f"Successfully made an intermediate save of {success_count} label images")
  print("")
  return success_count

def create_labels(model, x, img_data, DATA_LOC, subdir_names, test_mode = False, use_image_subdir = True):
  """
  Reads in our data and creates labels by running them through a forward pass
  """
  # Intialize return structure
  labels = []
  success_count = 0

  # Keep track of which img_data we need to pass to create_labels
  start_img_data = 0
  end_img_data = 0

  # Syncronize cuda (DISABLED FOR NOW)
  #torch.cuda.synchronize()

  # Create batches
  num_batches = math.ceil(x.shape[0]/ 20)
  
  # Split x into num_batches batches (can be different sizes)
  batches = np.array_split(x, num_batches, axis = 0)
  print("Creating Labels:")

  # If we are in test mode, let's only look at two batches
  if test_mode:
    print("Test mode is active: Only will compute two batches")
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
      
      # Clear Logits to conserve GPU RAM
      del logits

      # Append results
      labels.append(label_predictions)
      print(f"-Completed after {time.time() - start_time} seconds")

      # If we are at a 50th batch, let's save the labels and reset to
      # conserve CPU RAM
      if (i!=0 and i % 50 == 0) or i == len(batches) - 1:

        # concatenate the labels together
        labels_concat = np.concatenate(labels, axis = 0)

        # Update start_img_data and end_img_data for this batch
        # Update start img_data to the previous value of end image data
        start_img_data = end_img_data
        
        # Update end image data by the current batch group size
        end_img_data = end_img_data + labels_concat.shape[0]

        print(f"Starting intermediate save of {end_img_data - start_img_data} label images ...")

        # Split the img_data for this batch group
        split_0 = img_data[0][start_img_data:end_img_data]
        split_1 = img_data[1][start_img_data:end_img_data][:]
        img_data_split = (split_0, split_1)

        # Create the label images
        # Increment success count by the number of successful images saved
        success_count += create_label_images(labels_concat, img_data_split, DATA_LOC, subdir_names, use_image_subdir)

        # clear labels to conserve CPU RAM
        labels = []

        # delete labels_concat to conserve CPU RAM
        del labels_concat

  if test_mode:
    total_labels_test_mode = len(batches[0]) + len(batches[1])
    print(f"Successfully saved {success_count} out of {total_labels_test_mode} labels")
  else:
    print(f"Successfully saved {success_count} out of {x.shape[0]} labels")
  

def read_in_images(DATA_LOC: str, subdir_names = [], use_image_subdir = True):
  """
  Reads in images from the dataset location and returns a numpy array containing
  the images of shape (NxCxHxW), where N is the number of images, C is the
  number of channels, and H + W are sizes of spatial dimensions.
  """

  if not subdir_names:
    subdir_names = ['train_set', 'val_set', 'test_set']

  # Glob the data together
  X = []

  # Collect all *.jpg files in sub-directories
  file_list = []
  for name in subdir_names:
    files_at_name_folder = []
    if use_image_subdir:
      files_at_name_folder = glob.glob(os.path.join(DATA_LOC, name, "images", "*"))
    else:
      print("Not using image subdirectory in image path name")
      files_at_name_folder = glob.glob(os.path.join(DATA_LOC, name, "*.jpg"))
    file_list = file_list + files_at_name_folder

  # save the sizes of each file so we can resize at the end
  size_list = []
  print("Loading in images:")
  print(f"{len(file_list)} image files detected")
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