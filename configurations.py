from detectron2.config import get_cfg
from detectron2 import model_zoo


def load_cfg():
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.TEST.EVAL_PERIOD = 100

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.000025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1500    # 300 was good for balloon toy dataset. Adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for balloon toy dataset (default: 512)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (utrada tip). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.  

    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [1.0, 1.0]

    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # keep and do not exclude images labeled to have no objects

    return cfg