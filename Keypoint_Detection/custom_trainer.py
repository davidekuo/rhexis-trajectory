from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name):

    return COCOEvaluator(dataset_name, cfg)


class CustomAugmentation(CocoTrainer):

  @classmethod
  def build_train_loader(cls, cfg):
    augmentations = [
      T.RandomBrightness(.5,1.5),
      T.RandomContrast(.5,1.5),
      T.RandomSaturation(.5,1.5),
      T.RandomCrop('relative_range', (.9,.9)),
      T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ),
      T.RandomFlip(horizontal=True),
      T.RandomFlip(horizontal=False, vertical=True)
      ]
    mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
    return build_detection_train_loader(cfg, mapper=mapper)