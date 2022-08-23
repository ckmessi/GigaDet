#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/14 19:35
# @File     : detect_panda_image.py

"""
import argparse
import os
import time

import cv2
import torch

from gigadetect.core.panda_inference import PandaDetectService
from gigadetect.config import parser as decdet_parser
from pgn.config.cfg_node import get_cfg as get_default_pgn_cfg
from gigadetect.config.cfg_node import get_decdet_default_cfg
from gigadetect.schema.giga_detection import GigaDetection
from gigadetect.utils import visualizer, geometry
from gigadetect.utils.logger import DEFAULT_LOGGER


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to gigadet eval config file")
    parser.add_argument('--image_path', type=str, default='', help='the path of input image')

    parser.add_argument('--show-image', action='store_true',
                        help='display the detection results, do not use it if in command line env')
    parser.add_argument('--save_path', type=str, default=None,
                        help='the file path to store image with detection result')
    parser.add_argument('--annotation_file_path', type=str, default='', help='the file path of panda annotation.')

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


def post_process_display(args, image, giga_detections):
    display_image = args.show_image
    if display_image:
        visualizer.show_detection_result(image, [], giga_detections, target_size=1080, draw_label=False)

    save_path = args.save_path
    if save_path:
        image_with_detection = visualizer.draw_detection_result_to_image(image, giga_detections, draw_label=False)
        # Optional
        target_size = 3840
        ratio = target_size / max(image.shape[0], image.shape[1])
        interp = cv2.INTER_AREA
        image_resize = cv2.resize(image_with_detection, (int(image_with_detection.shape[1] * ratio), int(image_with_detection.shape[0] * ratio)),
                                  interpolation=interp)
        cv2.imwrite(save_path, image_resize)


def evaluate_pgn(self, image_path: str, image_origin, annotation_file_path: str, image_patch_list):
    labels_out = PandaDetectService.get_labels(image_path, annotation_file_path)
    label_boxes = PandaDetectService.build_gt_bbox_list(image_origin, labels_out)
    label_boxes_np = label_boxes.numpy()
    gt_bbox_list = [
        GigaDetection(label[0], label[1], label[2], label[3], 0, 1.0) for
        label in label_boxes_np]

    patch_bboxes = torch.Tensor([[patch.left_left, patch.top_top, patch.left_left+patch.width, patch.top_top+patch.height]for patch in image_patch_list])
    patch_bboxes_np = patch_bboxes.numpy()
    bbox_iou_of_bbox_labels = geometry.bbox_iou_of_bbox_b(patch_bboxes_np, label_boxes_np)

    max_iou_for_labels = bbox_iou_of_bbox_labels.max(axis=0)
    threshold = 0.7
    include_count = max_iou_for_labels[max_iou_for_labels>threshold].shape[0]
    pgn_recall = float(include_count) / labels_out.shape[0]
    print(f'pgn recall is {pgn_recall}')

    # display
    self.__visualize_giga_patch(image, image_patch_list, [])
    # Optional
    target_size = 3840
    ratio = target_size / max(image.shape[0], image.shape[1])
    interp = cv2.INTER_AREA
    image_resize = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=interp)
    cv2.imwrite("test.jpg", image_resize)


    return pgn_recall


if __name__ == '__main__':

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


    # load input image
    image_path = args.image_path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Error occurs when loading input image: `{image_path}`')

    panda_detect_service = PandaDetectService(gigadet_eval_cfg, pgn_cfg, decdet_cfg.MODEL.WEIGHTS,
                                              conf_threshold=decdet_cfg.TEST.DECDET.CONF_THRESHOLD,
                                              pgn_patch_top_k=gigadet_eval_cfg.TEST.PGN.PATCH_TOP_K,
                                              img_size=gigadet_eval_cfg.TEST.DECDET.INPUT_SIZE,
                                              pgn_patch_iou_threshold=gigadet_eval_cfg.TEST.PGN.PATCH_IOU_THRESHOLD)


    # image_patch_list = panda_detect_service.get_image_patches(image, patch_top_k=32)
    # pgn_recall = panda_detect_service.evaluate_pgn(image_path, image, args.annotation_file_path, image_patch_list)


    # execute inference
    t0 = time.time()
    giga_detections = panda_detect_service.detect_image(image)
    t1 = time.time()
    print(f'inference cost {t1 - t0}s.')

    # print detections info
    # [print(giga_detection) for giga_detection in giga_detections]
    DEFAULT_LOGGER.info(f'detect total {len(giga_detections)} persons.')

    # [optional] display/save annotated image
    post_process_display(args, image, giga_detections)

    # [optional] evaluate map for current single
    if args.annotation_file_path:
        panda_detect_service.evaluate(image_path, image, args.annotation_file_path, giga_detections)

    DEFAULT_LOGGER.info("The panda inference procedure is finished.")
