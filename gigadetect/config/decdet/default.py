#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/1/30 15:19
# @File     : default.py

"""

from gigadetect.config.cfg_node import CfgNode as CN

_C = CN()

_C.INFO = CN()
_C.INFO.CFG_NAME = 'decdet'
_C.INFO.UID = "default_uid"
_C.INFO.LOG_LEVEL = "INFO"

# model
_C.MODEL = CN()
_C.MODEL.WEIGHTS = ""
_C.MODEL.CFG_FILE_PATH = "gigadetect/models/yolov5/yolov5s.yaml"
_C.MODEL.ARCH = 'yolov5s'
# cuda device, i.e. 0 or 0,1,2,3 or cpu
_C.MODEL.CUDA_DEVICE = "0"

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.CLS_GAIN = 0.58
_C.MODEL.LOSS.BCE_CLS_POSITIVE_WEIGHT = 1.0 # cls BCELoss positive_weight
_C.MODEL.LOSS.BCE_OBJ_POSITIVE_WEIGHT = 1.0 # obj BCELoss positive_weight
_C.MODEL.LOSS.GIOU_GAIN = 0.05 # giou loss gain
_C.MODEL.LOSS.OBJ_GAIN = 1.0 # obj loss gain (*=img_size/320 if img_size != 320)
_C.MODEL.LOSS.GIOU_LOSS_RATIO = 1.0 # giou loss ratio (obj_loss = 1.0 or giou)

_C.MODEL.LOSS.FOCAL_LOSS = CN()
_C.MODEL.LOSS.FOCAL_LOSS.ENABLED = False
_C.MODEL.LOSS.FOCAL_LOSS.GAMMA = 0.0 # focal loss gamma (efficientDet default is gamma=1.5)

_C.MODEL.ANCHOR_THRESHOLD = 4.0 # anchor-multiple threshold
_C.MODEL.IOU_THRESHOLD = 0.2 # iou training threshold

# dataset
_C.DATASETS = CN()
_C.DATASETS.TRAIN_ROOT = "/path/to/train/"
_C.DATASETS.VAL_ROOT = "/path/to/val"
_C.DATASETS.CACHE_IMAGES = False
_C.DATASETS.RECT_TRAINING = False
_C.DATASETS.SINGLE_CLASS = False

_C.DATASETS.NUM_CLASSES = 2
_C.DATASETS.CLASS_NAMES = ["Category_1", "Category_2"]

# solver
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER = 'SGD'
# _C.SOLVER.LR_DECAY = 0.1
_C.SOLVER.BASE_LR = 0.01    # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.MOMENTUM = 0.937
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.MAX_EPOCH = 300
_C.SOLVER.SKIP_EVALUATE = False

# input
_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE_TRAIN = [640, 640]
_C.INPUT.IMAGE_SIZE_TEST = [640, 640]
_C.INPUT.MULTI_SCALE = False
_C.INPUT.ROTATE_DEGREES = 0.0  # image rotation (+/- deg)
_C.INPUT.TRANSLATE = 0.0 # image translation (+/- fraction)
_C.INPUT.SCALE = 0.5  # image scale (+/- gain)
_C.INPUT.SHEAR = 0.0 # image shear (+/- deg)
_C.INPUT.HSV_H = 0.014  # image HSV-Hue augmentation (fraction)
_C.INPUT.HSV_S = 0.68  # image HSV-Saturation augmentation (fraction)
_C.INPUT.HSV_V = 0.36  # image HSV-Value augmentation (fraction)

# test
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.DECDET = CN()
_C.TEST.DECDET.CONF_THRESHOLD = 0.001
_C.TEST.DECDET.NMS_IOU_THRESHOLD = 0.6
_C.TEST.DECDET.INPUT_SIZE = 320

_C.TEST.VERBOSE = False


# output
_C.OUTPUT = CN()
_C.OUTPUT.ROOT_DIR = 'outputs/'
