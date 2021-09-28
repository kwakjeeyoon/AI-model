import os

import albumentations
import detectorn2
import detectron2.utils.logger improt steup_logger
setup_logger()

# import some common detectron2 utilities (라이브러리 및 모듈 import 하기)
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# 데이터셋 등록
register_coco_instances("coco_trash_train", {}, '/home/data/data/train.json', '/home/data/data')
register_coco_instances("coco_trash_val", {}, '/home/data/data/val.json', '/home/data/data')

# config 파일 불러오기
cfg = get_cfg()     # get a copy of the default config
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

# config 수정하기
cfg.DATASETS.TRAIN = ("coco_trash_train",)
cfg.DATASETS.TEST = ("coco_trash_val",)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000 # epoch a = max_iter * batch_size / total_num_images
cfg.SOLVER.STEPS = (1000,1500) # iteration number to decrease learning rate by GAMMA
cfg.SOLVER.GAMMA = 0.05

# Augmentation mapper 정의
def MyMapper(dataset_dict):
    """Mapper which uses 'detectron2.data.transforms' augmentations"""
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
        T.RandomFlip(prob = 0.4, horizontal = False, vertical = True)
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    image, transfroms = T.apply_transform_gens(transform_list, image)
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd",0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

# Trainer 정의
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper = MyMapper, sampler = sampler
        )
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("./output_eval",exist_ok=True)
            output_folder = './output_eval'
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

# 학습
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # './output'
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
