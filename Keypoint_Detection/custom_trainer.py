from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T


class CocoTrainer(DefaultTrainer):
    """
    This class extends the DefaultTrainer class. Utilize this class to train a 
    model with no data augmentation.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        This function overrides the DefaultTrainer.build_evaluator function.
        """
        return COCOEvaluator(dataset_name, cfg)


class CustomAugmentation(CocoTrainer):
    """
    This class extends the CocoTrainer class. Utilize this class to train a
    model with data augmentation as defined in the build_train_loader function.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        """
        This function overrides the CocoTrainer.build_train_loader function with
        custom data augmentation.
        """
        augmentations = [
            T.RandomBrightness(0.5, 1.5),
            T.RandomContrast(0.5, 1.5),
            T.RandomSaturation(0.5, 1.5),
            T.RandomCrop("relative_range", (0.9, 0.9)),
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomFlip(horizontal=True),
            T.RandomFlip(horizontal=False, vertical=True),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)


class NoAugmentation(CocoTrainer):
    """
    This class extends the CocoTrainer class. Utilize this class to train a
    model with no data augmentation.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        """
        This function overrides the CocoTrainer.build_train_loader function with
        no data augmentation.
        """
        augmentations = []
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)
