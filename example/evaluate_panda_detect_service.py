#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/3 19:55
# @File     : evaluate_panda_detect_service.py

"""
import argparse

import time
import os
import torch
import numpy as np
from tqdm import tqdm

from gigadetect.core.panda_inference import PandaDetectService, output_evaluate_statistics
from pgn.config.cfg_node import get_cfg as get_default_pgn_cfg
from gigadetect.config.cfg_node import get_decdet_default_cfg
from pgn.datasets.panda import PandaImageAndLabelDataset
from gigadetect.schema.giga_detection import GigaDetection
from gigadetect.utils.logger import DEFAULT_LOGGER
from gigadetect.config import parser as decdet_parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to gigadet eval config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def set_specific_config_for_eval(pgn_cfg):
    pgn_cfg.defrost()
    pgn_cfg.DATASETS.PANDA.REDUCE_INPUT = False
    return pgn_cfg


if __name__ == "__main__":

    args = parse_args()

    # gigadet
    gigadet_eval_cfg = decdet_parser.setup_cfg_for_gigadet_eval(args, prepare_env=True)

    # pgn
    pgn_config_file = gigadet_eval_cfg.TEST.PGN.CFG_FILE_PATH
    if not os.path.exists(pgn_config_file):
        raise ValueError(f"pgn_config_file is invalid: {pgn_config_file}")
    pgn_cfg = get_default_pgn_cfg()
    pgn_cfg.merge_from_file(pgn_config_file)
    pgn_cfg = set_specific_config_for_eval(pgn_cfg)


    # decdet
    decdet_config_file = gigadet_eval_cfg.TEST.DECDET.CFG_FILE_PATH
    if not os.path.exists(pgn_config_file):
        raise ValueError(f"decdetet pgn_config_file is invalid: {pgn_config_file}")
    decdet_cfg = get_decdet_default_cfg()
    decdet_cfg.merge_from_file(decdet_config_file)

    # set model
    if gigadet_eval_cfg.TEST.PGN.MODEL.WEIGHTS:
        pgn_cfg.MODEL.WEIGHTS = gigadet_eval_cfg.TEST.PGN.MODEL.WEIGHTS
    if gigadet_eval_cfg.TEST.DECDET.MODEL.WEIGHTS:
        decdet_cfg.MODEL.WEIGHTS = gigadet_eval_cfg.TEST.DECDET.MODEL.WEIGHTS


    # load dataset
    dataset = PandaImageAndLabelDataset(pgn_cfg, pgn_cfg.DATASETS.DATASET_ROOT, split='test', target_key=pgn_cfg.DATASETS.PANDA.PERSON_KEY)
    batch_size = 1
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              pin_memory=False,
                                              collate_fn=PandaImageAndLabelDataset.collate_fn
                                              )

    # load service
    panda_detect_service = PandaDetectService(gigadet_eval_cfg, pgn_cfg, decdet_cfg.MODEL.WEIGHTS,
                                              conf_threshold=decdet_cfg.TEST.DECDET.CONF_THRESHOLD,
                                              pgn_patch_top_k=gigadet_eval_cfg.TEST.PGN.PATCH_TOP_K,
                                              img_size=gigadet_eval_cfg.TEST.DECDET.INPUT_SIZE,
                                              pgn_patch_iou_threshold=gigadet_eval_cfg.TEST.PGN.PATCH_IOU_THRESHOLD)

    statistics_list = []
    statistics_list_small = []
    statistics_list_medium = []
    statistics_list_large = []
    seen_object_count_total = 0
    seen_object_count_total_small = 0
    seen_object_count_total_medium = 0
    seen_object_count_total_large = 0
    inference_time_statistics_np = np.zeros(0)

    # DEFAULT_LOGGER.configure(handlers=[{"sink": sys.stderr, "level": args.log_level}])
    for i, (imgs, labels, imgs_origin) in tqdm(enumerate(data_loader)):
        # labels format is ltrb
        imgs = imgs.numpy()
        imgs_origin = imgs_origin.numpy()
        DEFAULT_LOGGER.debug(f"iterator index #{i}")
        DEFAULT_LOGGER.debug(f"image shape as {imgs.shape}, label shape as {labels.shape}")

        global_detections = []
        for index_in_batch in range(0, imgs.shape[0]):
            cur_labels = labels[labels[:, 0] == index_in_batch]
            h, w = imgs[index_in_batch].shape[0], imgs[index_in_batch].shape[1]

            t0 = time.time()
            if gigadet_eval_cfg.TEST.GIGADET.WITHOUT_ENCAPSULATE:
                outputs = panda_detect_service.detect_image_without_encapsulate(imgs_origin[index_in_batch], detect_directly=not gigadet_eval_cfg.TEST.GIGADET.SKIP_DETECT_DIRECTLY)
            else:
                global_detection_in_batch_sample = panda_detect_service.detect_image(imgs_origin[index_in_batch], detect_directly=not gigadet_eval_cfg.TEST.GIGADET.SKIP_DETECT_DIRECTLY)

            t1 = time.time()
            inference_time = t1 - t0
            if gigadet_eval_cfg.TEST.GIGADET.WITHOUT_ENCAPSULATE:
                global_detection_in_batch_sample = GigaDetection.buildGigaDetectionFromModelOutputs(outputs)

            DEFAULT_LOGGER.info(f'current detection contains {len(global_detection_in_batch_sample)} persons, cost {t1-t0} seconds')

            global_detections.append(global_detection_in_batch_sample)
            inference_time_statistics_np = np.append(inference_time_statistics_np, inference_time)
            DEFAULT_LOGGER.info(f'Current average inference time: {np.mean(inference_time_statistics_np)} s')

        epoch_stats, seen_object_count, \
        epoch_stats_small, seen_object_count_small, \
        epoch_stats_medium, seen_object_count_medium, \
        epoch_stats_large, seen_object_count_large = panda_detect_service.evaluate_single_image_batch(imgs_origin,
                                                                                                      global_detections,
                                                                                                      labels, show_every_image=gigadet_eval_cfg.TEST.GIGADET.SHOW_EVERY_IMAGE)

        statistics_list.extend(epoch_stats)
        seen_object_count_total += seen_object_count
        statistics_list_small.extend(epoch_stats_small)
        seen_object_count_total_small += seen_object_count_small
        statistics_list_medium.extend(epoch_stats_medium)
        seen_object_count_total_medium += seen_object_count_medium
        statistics_list_large.extend(epoch_stats_large)
        seen_object_count_total_large += seen_object_count_large

    output_evaluate_statistics('evaluation for `all` object.', statistics_list, seen_object_count_total, batch_size)

    output_evaluate_statistics('evaluation for `small` object.', statistics_list_small, seen_object_count_total_small, batch_size)

    output_evaluate_statistics('evaluation for `medium` object.', statistics_list_medium, seen_object_count_total_medium, batch_size)

    output_evaluate_statistics('evaluation for `large` object.', statistics_list_large, seen_object_count_total_large, batch_size)

    print("Hello")

