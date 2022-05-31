import numpy as np
from skimage.morphology import *
from google.colab.patches import cv2_imshow


def get_class_segmentation(label, class_int: int):
  """
  Returns a boolean mask of the requested class that matches class_int in label
  !Does not do any post processing!

  Parameters:
    label: A HXW numpy array of the image segmentation label

    class_int: An integer that corresponds to the class we would like to access

  Returns: A HxW boolean numpy array with True where label == class_int and
    False otherwise.
  """
  return label == class_int

def get_multiple_class_segmenation(label, class_integers):
  """
  Returns a boolean mask that includes all regions represented by integers in
  the class_integers list.
  !Does not do any post processing to segmented regions!

  Parameters:
    label: A HXW numpy array of the image segmentation label

    class_integers: A list of integers corresponding to the classes we would
      like a mask of.

  Returns: A HxW boolean numpy array with True where label had any of the int
    values inside of class_integers and False otherwise.
  """

  mask = label == -1
  for ci in class_integers:
    mask[label == ci] = True

  return mask

def get_class_int_from_name(name: str, label_dict):
  """
  Returns the integer that corresponds to the label name from the label_dict

  Parameters:
    name: String name of the class you would like to extract

    label_dict: The dict containing integer keys and string name values
    (access by calling segmentation_access_functions.get_labels(task))

  Returns: integer that indicates the class with the title 'name' in the labels
  """
  # Inverse the dict mapping
  inv_label_dict = {v: k for k, v in label_dict.items()}

  # Return the int
  return inv_label_dict[name]


def extract_pupil_filled(label, label_dict):
  """
  Applies post processing to the binary_mask of the pupil and returns filled in
  pupil mask

  Parameters:
    label: HxW numpy array that contains the segmented label of the image

    label_dict: The dict containing integer keys and string name values
    (access by calling segmentation_access_functions.get_labels(task))

  Returns: Binary mask corresponding to the filled pupil region
  """

  # Collect the class ints of Pupil
  pupil_int = get_class_int_from_name("Pupil", label_dict)

  # get Pupil and iris masks
  pupil_mask = get_class_segmentation(label, pupil_int)

  # Only keep large bodies in pupil mask
  binary_mask = getLargestCC(pupil_mask)

  # Apply convx_hull_image transform
  binary_mask = convex_hull_image(binary_mask)

  return binary_mask


def extract_pupil_median_pos(label, label_dict):
  """
  Returns median position of the pupil

  Parameters:
    label: HxW numpy array that contains the segmented label of the image

    label_dict: The dict containing integer keys and string name values
    (access by calling segmentation_access_functions.get_labels(task))

  Returns: List containing [median x pos, median y pos]
  """

  full_pupil = extract_pupil_filled(label, label_dict)

  pos = np.argwhere(full_pupil)

  median = np.median(pos, axis = 0)

  return [round(median[0]) , round(median[1])]

def place_dot(point_image, point_x, point_y):
  """
  Places a dot on for the display_dot_on_pos function
  """
  # dot_size
  dot_size = 5

  # correct for edges
  x_down = min(dot_size, point_x - 0)
  x_up = min(dot_size, point_image.shape[0] - point_x)

  y_down = min(dot_size, point_y - 0)
  y_up = min(dot_size, point_image.shape[1] - point_y)

  point_image[point_x-x_down:point_x+x_up, point_y-y_down:point_y+y_up, :] = [0,0,255]

  return point_image
  


def display_dot_on_pos(im_or_lab, pos):
  """ 
  Displays the median pupil position over image or label image

  Parameters:
    im_or_lab: HxW numpy array that contains the segmented label of the image
      OR an HxWx3 numpy array that contains the original image

    pos: A list of [x, y] lists containing positions we should place dots OR
      a list of two integers (x,y) to place a single point
  """
  point_image = []
  if len(im_or_lab.shape) == 3:
    point_image = im_or_lab
  else:
    point_image = np.stack([im_or_lab, im_or_lab, im_or_lab],axis=2)
  
  # determine if pos is a single point or a list of points
  if type(pos[0]) is int:
    # Single point
    point_x = pos[0]
    point_y = pos[1]
    point_image = place_dot(point_image, point_x, point_y)

  else:
    for point in pos:
      point_x = point[0]
      point_y = point[1]
      point_image = place_dot(point_image, point_x, point_y)

  cv2_imshow(point_image * 10)


def extract_pupil_extents(label, label_dict):
  """
  Extract spatial extents of the pupil.

  Parameters:
    label: HxW numpy array that contains the segmented label of the image

    label_dict: The dict containing integer keys and string name values
      (access by calling segmentation_access_functions.get_labels(task))

  Returns:
    Extents in list: (left, right, up, down)
  """

  # 
  full_pupil = extract_pupil_filled(label, label_dict)

  pos = np.argwhere(full_pupil)

  # Determine max x and max y values
  maxes = np.max(pos, axis = 0)

  # Determine min x and min y values
  mins = np.min(pos, axis = 0)


  # Determine median pupil value
  median = extract_pupil_median_pos(label, label_dict)

  # Generate extents
  left = [mins[0] , median[1]]
  right = [maxes[0] , median[1]]
  up = [median[0] ,maxes[1]]
  down = [median[0] ,mins[1]]

  return [left, right, up, down]


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def area_body_filtering(mask, thresh = 0.25):
  """
  Removes small bodies in a binary image by thresholding their area reletive to
  the total number of True pixels in the image.
  """
  area_threshold = np.count_nonzero(mask) * thresh
  return area_opening(mask, area_threshold)


def extract_incision_position(label, label_dict):
  """
  Extracts position of incision in image

  Parameters:
    label: HxW numpy array that contains the segmented label of the image

    label_dict: The dict containing integer keys and string name values
    (access by calling segmentation_access_functions.get_labels(task))

  Returns: A list of the [x pos, y pos] of the incision
  """

  # Collect the class ints of cornea, iris, and forceps
  cornea_int = get_class_int_from_name("Cornea", label_dict)
  iris_int = get_class_int_from_name("Iris", label_dict)
  forcep_int = get_class_int_from_name("Cap. Forceps", label_dict)

  # Collect masks of above classes
  cornea_mask = get_class_segmentation(label, cornea_int)
  iris_mask = get_class_segmentation(label, iris_int)
  forcep_mask = get_class_segmentation(label, forcep_int)

  # Apply area filtering
  cornea_mask = area_body_filtering(cornea_mask)
  iris_mask = area_body_filtering(iris_mask)
  forcep_mask = area_body_filtering(forcep_mask)

  
  for i in range(3):
    # Diolate individuals masks
    iris_mask = binary_dilation(iris_mask)
    forcep_mask = binary_dilation(forcep_mask)
    cornea_mask = binary_dilation(cornea_mask)

  # Determine regions that contain all three of these classes
  binary_mask = np.logical_and(cornea_mask, iris_mask)
  binary_mask = np.logical_and(binary_mask, forcep_mask)

  if binary_mask.any():
    # Determine median pos of incision
    pos = np.argwhere(binary_mask)
    median = np.median(pos, axis = 0)

    return [round(median[0]) , round(median[1])]
  else:
    return [-1, -1]


  
  




