from detectron2.config import get_cfg
from detectron2 import model_zoo


def load_cfg(model_string: str, MAX_ITER = 1500):
  """ Returns the config file for the specified model
  Parameters:
    model_string - String containing the model we would like to utilize

  Return: 
    cfg - The finished cfg file
  """
  cfg = get_cfg()

  model_string = model_string + ".yaml"

  valid_options = ["keypoint_rcnn_R_50_FPN_3x.yaml",
    "keypoint_rcnn_R_101_FPN_3x.yaml",
    "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"]

  if model_string not in valid_options:
    assert(False)

  cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Keypoints/{model_string}"))

  cfg.DATASETS.TRAIN = ("train",)
  cfg.DATASETS.TEST = ("val",)
  cfg.TEST.EVAL_PERIOD = 100

  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Keypoints/{model_string}")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.000025  # pick a good LR
  cfg.SOLVER.MAX_ITER = MAX_ITER    # 300 was good for balloon toy dataset. Adjust up if val mAP is still rising, adjust down if overfit
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for balloon toy dataset (default: 512)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (utrada tip). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
  # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.  

  cfg.MODEL.KEYPOINT_ON = True
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
  cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.03, 0.03]

  cfg.SOLVER.CHECKPOINT_PERIOD = 500
  # tells Detectron2 to save a model checkpoint every X iterations
  # checkpoints are saved as 'model_{iteration_number}.pth
  # Detectron2 also creates a file 'last_checkpoint' which simply contains the filename of .pth file for the last checkpoint (ex. model_0000079.pth)
  # To resume training from last_checkpoint, Detectron2 needs 'last_checkpoint' and the corresponding .pth file in cfg.OUTPUT_DIR

  # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # keep and do not exclude images labeled to have no objects

  return cfg