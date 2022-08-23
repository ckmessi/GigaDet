#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/2/5 17:53
# @File     : gigadet_eval_default.py

"""

from gigadetect.config.cfg_node import CfgNode as CN

_C = CN()

_C.INFO = CN()
_C.INFO.UID = "gigadet_eval_default"
_C.INFO.LOG_LEVEL = "INFO"


# test
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.VERBOSE = False

# test-pgn
_C.TEST.PGN = CN()
_C.TEST.PGN.CFG_FILE_PATH = '/path/to/pgn/cfg/file'
_C.TEST.PGN.PATCH_TOP_K = 64
_C.TEST.PGN.PATCH_IOU_THRESHOLD = 0.2
_C.TEST.PGN.MODEL = CN()
_C.TEST.PGN.MODEL.WEIGHTS = '/path/to/pgn/model'

# test-decdet
_C.TEST.DECDET = CN()
_C.TEST.DECDET.CFG_FILE_PATH = '/path/to/decdet/cfg/file'
_C.TEST.DECDET.MODEL = CN()
_C.TEST.DECDET.MODEL.WEIGHTS = '/path/to/decdet/model'
_C.TEST.DECDET.INPUT_SIZE = 320


# test-gigadet
_C.TEST.GIGADET = CN()
_C.TEST.GIGADET.SHOW_EVERY_IMAGE = True
_C.TEST.GIGADET.WITHOUT_ENCAPSULATE = True
_C.TEST.GIGADET.SKIP_DETECT_DIRECTLY = True
# gigadet-exp-random-split-grid
_C.TEST.GIGADET.PLAIN_GRID = CN()
_C.TEST.GIGADET.PLAIN_GRID.ENABLED = False
_C.TEST.GIGADET.PLAIN_GRID.WIDTH = 1080
_C.TEST.GIGADET.PLAIN_GRID.HEIGHT = 1080
_C.TEST.GIGADET.PLAIN_GRID.OVERLAP_RATIO = 0.2
# directly split whole image to `K` grids
_C.TEST.GIGADET.K_GRID = CN()
_C.TEST.GIGADET.K_GRID.ENABLED = False

# output
_C.OUTPUT = CN()
_C.OUTPUT.ROOT_DIR = 'outputs/gigadet_eval/default/'
