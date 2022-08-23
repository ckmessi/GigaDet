#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/3 18:53
# @File     : panda_inference.py

"""
import json
from typing import List

import os
import numpy as np

import time
import torch

from gigadetect.core import post_process, evaluater
from gigadetect.core.evaluater import EvaluateStatistics
from gigadetect.core.decdet_inference import DecDetService
from gigadetect.datasets.panda import convert_to_label_ltrb_list, HEAD_KEY, VISIBLE_BODY_KEY
from gigadetect.schema.area_range import AREA_SMALL, AREA_MEDIUM, AREA_LARGE
from gigadetect.schema.giga_detection import GigaDetection, GigaPatch
from gigadetect.utils import visualizer, coordinate
from gigadetect.utils.logger import DEFAULT_LOGGER
from pgn.core.detect import detect_single as fetch_pgn_proposals
from pgn.core.model_loader import build_pgn_trainer
from gigadetect.utils import misc


class PandaDetectService(DecDetService):

    pgn_patch_top_k: int

    def __init__(self, gigadet_eval_cfg, pgn_cfg, model_path, img_size=640, conf_threshold=0.4, iou_threshold=0.5, device='', pgn_patch_top_k=128, pgn_patch_iou_threshold=0.2):
        super().__init__(model_path, img_size, conf_threshold, iou_threshold, device)
        self.gigadet_eval_cfg = gigadet_eval_cfg
        self.pgn_cfg = pgn_cfg
        self.__init_pgn_model(pgn_patch_top_k, pgn_patch_iou_threshold)

    def __init_pgn_model(self, pgn_patch_top_k, pgn_patch_iou_threshold):

        if self.pgn_cfg.MODEL.WEIGHTS:
            self.pgn_trainer = build_pgn_trainer(self.pgn_cfg)
            self.pgn_patch_top_k = pgn_patch_top_k
            self.pgn_patch_iou_threshold = pgn_patch_iou_threshold
        else:
            DEFAULT_LOGGER.warning(f"PGN Model has no pretrained model. Please check.")
            self.pgn_trainer = None
            self.pgn_patch_top_k = -1
            self.pgn_patch_iou_threshold = 0.2

    def detect_image(self, input_image, detect_directly=True) -> List[GigaDetection]:

        # from input_image get patch
        t0 = time.time()
        image_patch_list = self.get_image_patches(input_image, self.pgn_patch_top_k, self.pgn_patch_iou_threshold)
        t1 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #01 patch generate: {t1- t0}')

        # [DEBUG] show pgn result
        # self.__visualize_giga_patch(input_image, image_patch_list)

        giga_detections_list = []
        for patch_index in range(0, len(image_patch_list), 128):
            cur_image_patch_list = image_patch_list[patch_index:patch_index + 128]
            # detect all patches
            image_patch_data_list = [g.data for g in cur_image_patch_list]
            # [cv2.imwrite(str(i) + ".jpg", image_patch) for (i, image_patch) in enumerate(image_patch_data_list)]
            cur_giga_detections_list = self.detect_images(image_patch_data_list)
            giga_detections_list.extend(cur_giga_detections_list)

        t2 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #02 giga detection list fetched: {t2- t0}')


        filtered_giga_detections_list = GigaDetection.filterOnlyPersonCategory(giga_detections_list)

        t3 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #03 filter only person category: {t3- t0}')
        # remap to original image
        global_detection_list = []
        for (i, giga_patch) in enumerate(image_patch_list):
            current_global_detections = giga_patch.convert_to_global_detection_absolute_value(
                filtered_giga_detections_list[i])
            global_detection_list.extend(current_global_detections)

        t4 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #04 remap to original image: {t4- t0}')

        # detect on resized image
        if detect_directly:
            direct_detection_list = super().detect_image(input_image)
            direct_detection_list = GigaDetection.filterOnlyPersonCategory([direct_detection_list])
            global_detection_list.extend(direct_detection_list[0])
            t5 = time.time()
            DEFAULT_LOGGER.debug(f'[PANDA Inference] #05 detect directly: {t5- t0}')

        device = misc.get_target_device()
        output = [
            torch.FloatTensor([gd.convert_to_ltrb_confidence_category() for gd in giga_detections_list_one]).to(
                device) for
            giga_detections_list_one in [global_detection_list]
        ]
        outputs = post_process.non_max_suppression_for_batch_outputs(output, conf_thres=self.conf_threshold, iou_thres=0.6,
                                                                    merge=False)
        t6 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #06 final nms: {t6- t0}')

        global_detection_list = GigaDetection.buildGigaDetectionFromModelOutputs(outputs)
        t7 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #07 build outputs: {t7- t0}')

        return global_detection_list

    def detect_image_without_encapsulate(self, input_image, detect_directly=True, only_detect_directly=False):
        """
            Implement the same feature as `detect_image`.
            Improve inference efficiency while reducing readability.
        :param input_image:
        :param detect_directly:
        :return:
        """

        ### Only 20201031
        if only_detect_directly:
            direct_detection_list = self.detect_images_without_encapsulate([input_image])
            output = direct_detection_list
            outputs = post_process.non_max_suppression_for_batch_outputs(output, conf_thres=self.conf_threshold,
                                                                         iou_thres=0.6,
                                                                         merge=False)
            return outputs


        # from input_image get patch
        t0 = time.time()
        image_patch_list = self.get_image_patches(input_image, self.pgn_patch_top_k, self.pgn_patch_iou_threshold)
        t1 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #01 generate {len(image_patch_list)} patches: {t1- t0}')


        giga_detections_list = []
        for patch_index in range(0, len(image_patch_list), 128):
            cur_image_patch_list = image_patch_list[patch_index:patch_index + 128]
            # detect all patches
            image_patch_data_list = [g.data for g in cur_image_patch_list]
            # [cv2.imwrite(str(i) + ".jpg", image_patch) for (i, image_patch) in enumerate(image_patch_data_list)]
            cur_giga_detections_list = self.detect_images_without_encapsulate(image_patch_data_list)
            giga_detections_list.extend(cur_giga_detections_list)

        t2 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #02 giga detection list fetched: {t2- t0}')

        if False:
            # NOTE: assume detect results only contains specific category
            filtered_giga_detections_list = GigaDetection.filterOnlyPersonCategory(giga_detections_list)
        else:
            filtered_giga_detections_list = giga_detections_list

        t3 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #03 filter only person category: {t3- t0}')
        # remap to original image
        global_detection_list = torch.Tensor()
        for (i, giga_patch) in enumerate(image_patch_list):
            # current_global_detections = giga_patch.convert_to_global_detection_absolute_value(
            #     filtered_giga_detections_list[i])
            current_global_detections = giga_patch.convert_to_global_detection_absolute_value_without_encapsulate(filtered_giga_detections_list[i])
            # global_detection_list.extend(current_global_detections)
            global_detection_list = torch.cat((global_detection_list, current_global_detections), 0)

        t4 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #04 remap to original image: {t4- t0}')

        # [optional] detect on resized original image
        if detect_directly:
            direct_detection_list = self.detect_images_without_encapsulate([input_image])
            # direct_detection_list = GigaDetection.filterOnlyPersonCategory([direct_detection_list])
            global_detection_list = torch.cat((global_detection_list, torch.Tensor(direct_detection_list[0])), 0)
            t5 = time.time()
            DEFAULT_LOGGER.debug(f'[PANDA Inference] #05 detect directly: {t5- t0}')

        # device = misc.get_target_device()
        # output = [
        #     torch.FloatTensor([gd.convert_to_ltrb_confidence_category() for gd in giga_detections_list_one]).to(
        #         device) for
        #     giga_detections_list_one in [global_detection_list]
        # ]
        output = [global_detection_list]
        outputs = post_process.non_max_suppression_for_batch_outputs(output, conf_thres=self.conf_threshold, iou_thres=0.6,
                                                                     merge=False)
        t6 = time.time()
        DEFAULT_LOGGER.debug(f'[PANDA Inference] #06 final nms: {t6- t0}')
        return outputs


    def get_image_patches(self, input_image, patch_top_k=128, pgn_patch_iou_threshold=0.2):
        """
        :param input_image:
        :return:
        """
        if self.gigadet_eval_cfg.TEST.GIGADET.PLAIN_GRID.ENABLED:
            # For experiments
            w = self.gigadet_eval_cfg.TEST.GIGADET.PLAIN_GRID.WIDTH
            h = self.gigadet_eval_cfg.TEST.GIGADET.PLAIN_GRID.HEIGHT
            overlap_ratio = self.gigadet_eval_cfg.TEST.GIGADET.PLAIN_GRID.OVERLAP_RATIO
            grid_list = self.__split_to_grids(input_image, target_w=w, target_h=h, overlap_ratio=overlap_ratio)
            return grid_list[0:patch_top_k]
        elif self.gigadet_eval_cfg.TEST.GIGADET.K_GRID.ENABLED:
            # For experiments
            k = patch_top_k
            grid_list = self.__split_to_k_grids(input_image, k)
            assert len(grid_list) == k, f"Length error in split_to_k_grids."
            return grid_list
        else:
            if self.pgn_trainer:
                # use PGN to generate
                return self.__split_to_patch_by_pgn(input_image, top_k=patch_top_k, pgn_patch_iou_threshold=pgn_patch_iou_threshold)
            else:
                return self.__split_to_grids(input_image, target_w=1600, target_h=1600)

    def __split_to_patch_by_pgn(self, input_image, top_k=128, pgn_patch_iou_threshold=0.2):
        anchor_top_k, patch_count_top_k = fetch_pgn_proposals(self.pgn_trainer, input_image, top_k=top_k, pgn_patch_iou_threshold=pgn_patch_iou_threshold)

        origin_h, origin_w = input_image.shape[0], input_image.shape[1]
        patch_list = []
        for (i, anchor) in enumerate(anchor_top_k):
            patch = input_image[anchor[0]:anchor[2], anchor[1]:anchor[3]]
            giga_patch = GigaPatch(patch, origin_h, origin_w, anchor[2] - anchor[0], anchor[3] - anchor[1], anchor[1],
                                   anchor[0], possible_score=float(patch_count_top_k[i]))
            patch_list.append(giga_patch)

        return patch_list

    def __split_to_grids(self, input_image, target_w, target_h, overlap_ratio=1.0):
        grid_list = []
        h, w = input_image.shape[0], input_image.shape[1]
        left = 0
        top = 0
        while left < w:
            right = min(w, left + int(target_w))
            while top < h:
                bottom = min(h, top + int(target_h))
                # generate patch
                patch = input_image[(bottom - target_h):bottom, (right - target_w):right]
                img_h = bottom - top
                img_w = right - left
                giga_patch = GigaPatch(patch, h, w, img_h, img_w, left, top)
                grid_list.append(giga_patch)

                top = top + int(target_h * (1 - overlap_ratio))
            left = left + int(target_w * (1 - overlap_ratio))
            top = 0
        return grid_list

    def __split_to_k_grids(self, input_image, k):
        assert k in [8, 16, 32, 45, 64, 128], f"pgn top_k {k} is not supported for split_to_k_grids"
        row, column = {
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
            45: (5, 9),
            64: (8, 8),
            128: (8, 16)
        }[k]
        grid_list = []
        h, w = input_image.shape[0], input_image.shape[1]
        grid_width = int(w / column)
        grid_height = int(h / row)
        for i in range(0, row):
            for j in range(0, column):
                patch = input_image[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
                giga_patch = GigaPatch(patch, h, w, grid_height, grid_width, j*grid_width, i*grid_height)
                grid_list.append(giga_patch)
        return grid_list


    def __visualize_giga_patch(self, image, giga_patch_list: List[GigaPatch], gt_bbox_list: List[GigaDetection]):
        giga_detections = [
            GigaDetection(g.left_left, g.top_top, g.left_left + g.width, g.top_top + g.height, 0, g.possible_score) for
            g in giga_patch_list]
        visualizer.show_detection_result(image, giga_detections, gt_bbox_list, target_size=1080, draw_label=False)

    def evaluate(self, image_path: str, image_origin, annotation_file_path: str, giga_detections: List[GigaDetection]):
        labels_out = self.get_labels(image_path, annotation_file_path)
        return self.evaluate_single_image_batch(np.expand_dims(image_origin, 0), [giga_detections], labels_out)

    @staticmethod
    def get_labels(image_path, annotation_file_path: str, target_key=VISIBLE_BODY_KEY):
        if not os.path.exists(annotation_file_path):
            msg = f'[PANDA] Invalid annotation file path: {annotation_file_path}'
            DEFAULT_LOGGER.error(msg)
            raise ValueError(msg)

        category_name_index_dict = {
            'person': 0
        }
        file_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        scene_name = os.path.basename(dir_name)
        image_name = scene_name + "/" + file_name

        with open(annotation_file_path, 'r') as annotation_file:
            annotation_dict = json.load(annotation_file)
            image_annotation = annotation_dict[image_name]

            objects_list = image_annotation['objects list']
            label_list_ltrb = convert_to_label_ltrb_list(objects_list, category_name_index_dict, target_key=target_key)
            labels = torch.from_numpy(np.array(label_list_ltrb))
            number_labels = len(labels)
            labels_out = torch.zeros((number_labels, 6))
            if number_labels:
                labels_out[:, 1:] = labels
        return labels_out

    def evaluate_single_image_batch(self, imgs_origin, giga_detections_list, labels, show_every_image=False):
        device = misc.get_target_device()
        # calculate metrics for current image
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        output = [
            torch.FloatTensor([gd.convert_to_ltrb_confidence_category() for gd in global_detection_in_batch_sample]).to(
                device) for
            global_detection_in_batch_sample in giga_detections_list
        ]

        output = post_process.non_max_suppression_for_batch_outputs(output, conf_thres=0.001, iou_thres=0.6,
                                                                    merge=False)

        targets_ltrb = torch.tensor(labels)
        targets_xywh = targets_ltrb
        # labels shape is batch_size x 6，the 6 items represents `batch_index`, `category`, `l`, `t`, `r`, `b`
        targets_xywh[:, 2:6] = coordinate.ltrb2xywh(targets_ltrb[:, 2:6])
        nb, origin_height, origin_width, _ = imgs_origin.shape  # batch size, channels, height, width
        whwh = torch.Tensor([origin_width, origin_height, origin_width, origin_height]).to(device)
        epoch_stats, seen_object_count = evaluater.calculate_metric(output, targets_xywh.to(device), niou, iouv,
                                                                    origin_height,
                                                                    origin_width, device, whwh.to(device))


        # add small, medium, large
        epoch_stats_small, seen_object_count_small = evaluater.calculate_metric(output, targets_xywh.to(device), niou,
                                                                                iouv, origin_height,
                                                                                origin_width, device, whwh.to(device),
                                                                                area_range=AREA_SMALL)
        epoch_stats_medium, seen_object_count_medium = evaluater.calculate_metric(output, targets_xywh.to(device), niou,
                                                                                  iouv, origin_height,
                                                                                  origin_width, device, whwh.to(device),
                                                                                  area_range=AREA_MEDIUM)
        epoch_stats_large, seen_object_count_large = evaluater.calculate_metric(output, targets_xywh.to(device), niou,
                                                                                iouv, origin_height,
                                                                                origin_width, device, whwh.to(device),
                                                                                area_range=AREA_LARGE)

        if show_every_image:
            # epoch indexes
            output_evaluate_statistics('All', epoch_stats, seen_object_count, batch_size=1)
            output_evaluate_statistics('Small', epoch_stats_small, seen_object_count_small, batch_size=1)
            output_evaluate_statistics('Medium', epoch_stats_medium, seen_object_count_medium, batch_size=1)
            output_evaluate_statistics('Large', epoch_stats_large, seen_object_count_large, batch_size=1)

            # display image
            label_boxes = self.build_gt_bbox_list(imgs_origin[0], labels)
            label_boxes_np = label_boxes.numpy()
            gt_bbox_list = [
                GigaDetection(label[0], label[1], label[2], label[3], 0, 1.0) for
                label in label_boxes_np]
            # visualizer.show_detection_result(imgs_origin[0], gt_bbox_list, giga_detections_list[0], target_size=1080, draw_label=False)

        return epoch_stats, seen_object_count, \
               epoch_stats_small, seen_object_count_small, \
               epoch_stats_medium, seen_object_count_medium, \
               epoch_stats_large, seen_object_count_large


    @staticmethod
    def build_gt_bbox_list(image_origin, labels_out):
        origin_height, origin_width = image_origin.shape[0], image_origin.shape[1]
        whwh = torch.Tensor([origin_width, origin_height, origin_width, origin_height])
        index_in_batch = 0
        labels = labels_out[labels_out[:, 0] == index_in_batch, 1:]
        label_boxes = labels[:, 1:] * whwh
        return label_boxes


def output_evaluate_statistics(title, statistics_list, image_count, batch_size):
    DEFAULT_LOGGER.info(title)
    evaluate_statistics_large = EvaluateStatistics()
    evaluate_statistics_large.image_count = image_count
    evaluate_statistics_large.compute_statistics(statistics_list, 1)
    evaluate_statistics_large.print_results(statistics_list, "person", True, 0, 0, 640, batch_size, True)

