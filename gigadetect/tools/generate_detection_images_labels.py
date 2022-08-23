#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @File     : generate_detection_images_labels.py

"""
import argparse

import os
import random

import cv2
import torch
from tqdm import tqdm
import numpy as np

from gigadetect.core.panda_inference import PandaDetectService
from gigadetect.datasets.panda import FULL_BODY_KEY
from gigadetect.schema.giga_detection import GigaPatch, GigaDetection
from gigadetect.utils import geometry
from pgn.core.model_loader import build_pgn_trainer
from gigadetect.utils.logger import DEFAULT_LOGGER
from pgn.config.cfg_node import get_cfg as get_default_pgn_cfg
from pgn.core.detect import detect_single as fetch_pgn_proposals


def uniform_patch_uid(img_name, patch_index):
    return img_name + "-" + str(patch_index)


def patch_image_file_path(output_image_dir, img_name, patch_index):
    pure_image_name, ext = os.path.splitext(img_name)
    return os.path.join(output_image_dir, uniform_patch_uid(pure_image_name, patch_index) + ".jpg")


def patch_annotation_file_path(output_label_dir, img_name, patch_index):
    pure_image_name, ext = os.path.splitext(img_name)
    return os.path.join(output_label_dir, uniform_patch_uid(pure_image_name, patch_index) + ".txt")


def resize_img_if_it_is_too_large(img, max_size=768):
    size = max(img.shape[0], img.shape[1])
    if size < max_size:
        return img

    ratio = float(max_size) / size
    img_resized = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img_resized


def generate_patch_detection_annotation(image_patch: GigaPatch, image_origin, labels, image_name: str, patch_index,
                                        output_dir='yolo_label/'):
    # prepare dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_image_dir = os.path.join(output_dir, "images/")
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    output_label_dir = os.path.join(output_dir, "labels/")
    if not os.path.exists(output_label_dir):
        os.mkdir(output_label_dir)

    # save annotation
    label_boxes = PandaDetectService.build_gt_bbox_list(image_origin, labels)
    label_boxes_np = label_boxes.numpy()
    patch_bboxes = torch.Tensor(
        [[patch.left_left, patch.top_top, patch.left_left + patch.width, patch.top_top + patch.height] for patch in
         [image_patch]])
    patch_bboxes_np = patch_bboxes.numpy()
    bbox_iou_of_bbox_labels = geometry.bbox_iou_of_bbox_b(patch_bboxes_np, label_boxes_np)

    max_iou_for_labels = bbox_iou_of_bbox_labels.max(axis=0)
    label_boxes_in_patch = label_boxes[max_iou_for_labels >= 1.0]

    if label_boxes_in_patch.shape[0] <= 0:
        return

    # rel_h, rel_w = abs_h - top_top, abs_w - left_left
    label_boxes_in_patch_relative = torch.Tensor([
        [b[0] - image_patch.left_left, b[1] - image_patch.top_top, b[2] - image_patch.left_left,
         b[3] - image_patch.top_top]
        for b in label_boxes_in_patch
    ])

    # rat_h, rat_w = rel_h, rel_w / (patch_h, patch_w)
    # xywh
    label_boxes_in_patch_relative_ratio = torch.Tensor([
        [(b[0] + b[2]) / 2 / image_patch.width, (b[1] + b[3]) / 2 / image_patch.height,
         (b[2] - b[0]) / image_patch.width, (b[3] - b[1]) / image_patch.height]
        for b in label_boxes_in_patch_relative
    ])

    label_content = ""
    for b in label_boxes_in_patch_relative_ratio:
        line = "0 "
        line += " ".join([str(x.numpy()) for x in b])
        line += "\n"
        label_content += line

    # save label file
    save_label_path = patch_annotation_file_path(output_label_dir, image_name, patch_index)
    with open(save_label_path, 'w') as f:
        f.write(label_content)

    # save image file
    save_image_path = patch_image_file_path(output_image_dir, image_name, patch_index)
    img_data = resize_img_if_it_is_too_large(image_patch.data)
    cv2.imwrite(save_image_path, img_data)

    DEFAULT_LOGGER.info(f"Finish generate detection annotation for {image_name}-{patch_index}")


def _generate_patch_from_small_object(labels, image):
    # filter small_labels
    H, W = image.shape[0], image.shape[1]
    small_labels = []
    for label in labels:
        absolute_h = (label[5] - label[3]) * H
        if absolute_h <= 288:
            small_labels.append(label.numpy())

    if not small_labels:
        return []
    small_labels = np.array(small_labels)
    # focus on the small_labels, expand a area
    image_patch_list = []
    for small_label in small_labels:
        # label format is [index, category, l, t, r, b]
        center_x = int((small_label[2] + (small_label[4] - small_label[2]) / 2) * W)
        center_y = int((small_label[3] + (small_label[5] - small_label[3]) / 2) * H)

        target_w = random.randint(600, 2500)
        target_h = target_w
        offset_x = float(random.randint(0, 6) - 3) / 10 * target_w
        offset_y = float(random.randint(0, 6) - 3) / 10 * target_h

        patch_left = max(0, int(center_x + offset_x - target_w / 2))
        patch_right = min(int(center_x + offset_x + target_w / 2), W - 1)
        patch_top = max(0, int(center_y + offset_y - target_h / 2))
        patch_bottom = min(int(center_y + offset_y + target_h / 2), H - 1)

        patch_data = image[patch_top: patch_bottom, patch_left: patch_right]

        giga_patch = GigaPatch(patch_data, H, W, patch_bottom - patch_top, patch_right - patch_left, patch_left,
                               patch_top)
        image_patch_list.append(giga_patch)

    return image_patch_list


def split_to_patch_by_pgn(pgn_trainer, input_image, top_k=128, pgn_patch_iou_threshold=0.2):
    # TODO: logic same in panda_inference.py

    anchor_top_k, patch_count_top_k = fetch_pgn_proposals(pgn_trainer, input_image, top_k=top_k,
                                                        pgn_patch_iou_threshold=pgn_patch_iou_threshold)

    origin_h, origin_w = input_image.shape[0], input_image.shape[1]
    patch_list = []
    for (i, anchor) in enumerate(anchor_top_k):
        patch = input_image[anchor[0]:anchor[2], anchor[1]:anchor[3]]
        giga_patch = GigaPatch(patch, origin_h, origin_w, anchor[2] - anchor[0], anchor[3] - anchor[1], anchor[1],
                               anchor[0], possible_score=float(patch_count_top_k[i]))
        patch_list.append(giga_patch)

    return patch_list


def generate_panda_detection_annotation(pgn_trainer, image_path, annotation_file_path,
                                        target_key, generate_focus_small_object=False):
    # read image
    img_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Error occurs when loading input image: `{image_path}`')

    # read annotation
    print(f'target_key={target_key}')
    labels = PandaDetectService.get_labels(image_path, annotation_file_path, target_key=target_key)

    # fetch pgn_patch
    if not generate_focus_small_object:
        image_patch_list = split_to_patch_by_pgn(pgn_trainer, image)
    else:
        image_patch_list = _generate_patch_from_small_object(labels, image)

    # iterate each patch
    print(f"length of image_patch_list: {len(image_patch_list)}")
    for (patch_index, image_patch) in tqdm(enumerate(image_patch_list)):
        generate_patch_detection_annotation(image_patch, image, labels, img_name, patch_index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default='', help='image root dir')
    parser.add_argument('--annotation_file_path', type=str, default='', help='the file path of panda annotation.')
    parser.add_argument('--generate_focus_small_object', action='store_true')
    parser.add_argument('--panda_person_key', type=str, default='visible body', help='{visible body/full body/head}')
    parser.add_argument('--config-file', type=str, default='', help='')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # load pre-trained pgn_trainer
    pgn_cfg = get_default_pgn_cfg()
    pgn_cfg.merge_from_file(args.config_file)
    pgn_cfg.merge_from_list(args.opts)
    pgn_trainer = build_pgn_trainer(pgn_cfg)

    # check path
    image_root = args.image_root
    if not os.path.exists(image_root):
        raise ValueError(f'Invalid image root: `{image_root}`')
    annotation_file_path = args.annotation_file_path
    if not os.path.exists(annotation_file_path):
        raise ValueError(f'Invalid annotation file path: `{annotation_file_path}`')

    image_path_list = []
    scene_list = os.listdir(image_root)
    for scene_name in scene_list:
        scene_dir_path = os.path.join(image_root, scene_name)
        image_name_list = os.listdir(scene_dir_path)
        for image_name in image_name_list:
            image_path = os.path.join(image_root, scene_name, image_name)
            image_path_list.append(image_path)

    DEFAULT_LOGGER.info("Finish collect all PANDA images")
    DEFAULT_LOGGER.info(f"start to generate, generate_focus_small_object={args.generate_focus_small_object}")
    for image_path in tqdm(image_path_list):
        generate_panda_detection_annotation(pgn_trainer, image_path,
                                            annotation_file_path, target_key=args.panda_person_key,
                                            generate_focus_small_object=args.generate_focus_small_object)

    DEFAULT_LOGGER.info("The panda generate detection annotation procedure is finished.")
