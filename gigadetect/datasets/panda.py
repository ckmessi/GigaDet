#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/20 19:26
# @File     : panda.py

"""
import json
from pathlib import Path

import os
from typing import List, Dict

import cv2
import torch
from torch.utils.data import Dataset

from gigadetect.config import const
from gigadetect.core import pre_process
from gigadetect.utils.logger import DEFAULT_LOGGER
import numpy as np

VISIBLE_BODY_KEY = 'visible body'
HEAD_KEY = 'head'
FULL_BODY_KEY = 'full body'

def convert_to_label_ltrb_list(objects_list, category_name_index_dict, target_key = VISIBLE_BODY_KEY):
    label_ltrb_list = []
    for object in objects_list:
        category = object['category']
        if category not in category_name_index_dict:
            continue

        if category == "person":
            bbox = object['rects'][target_key]
        else:
            bbox = object['rect']

        left = bbox['tl']['x']
        top = bbox['tl']['y']
        right = bbox['br']['x']
        bottom = bbox['br']['y']
        label_ltrb_list.append([category_name_index_dict[category], left, top, right, bottom])
    return label_ltrb_list

class PandaImage:
    image_id: int
    image_name: str
    image_path: str
    image_width: int
    image_height: int
    objects_list: List[object]
    label_list: List

    def __init__(self, image_path, image_id=None, image_name=None, image_width=None, image_height=None):
        self.image_path = image_path
        self.image_id = image_id
        self.image_name = image_name
        self.image_width = image_width
        self.image_height = image_height
        self.objects_list = []
        self.label_list = []




class PandaImageAndLabelDataset(Dataset):

    category_name_index_dict: Dict[str, int]
    image_list: List[PandaImage]

    def __init__(self, image_root='image_train', annotation_file_path='person_bbox_train.json', target_key=VISIBLE_BODY_KEY):
        self.__initialize_category()
        self.__collect_images(image_root)
        self.__collect_labels(annotation_file_path, target_key=target_key)


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        """
        target_image
        :param index:
        :return:
        """
        image_path = self.image_list[index].image_path
        img_origin = cv2.imread(image_path)
        # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if img_origin is None:
            DEFAULT_LOGGER.error(f"[PANDA] Can not open image: {image_path}")

        # resize
        img_size = 640
        original_height, original_width = img_origin.shape[0], img_origin.shape[1]
        DEFAULT_LOGGER.debug(f'image_path: {image_path}, original_height: {original_height}, original_width: {original_width}')
        r = img_size / max(original_height, original_width)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img_origin, (int(original_width * r), int(original_height * r)), interpolation=interp)
        # img, ratio, pad = pre_process.add_letterbox_for_image(img, img_size, auto=False, scaleup=False)

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img_origin = np.ascontiguousarray(img_origin)
        img_origin = torch.from_numpy(img_origin)

        # label
        label_list_ltrb = self.image_list[index].label_list
        labels = torch.from_numpy(np.array(label_list_ltrb))
        number_labels = len(labels)
        labels_out = torch.zeros((number_labels, 6))
        if number_labels:
            labels_out[:, 1:] = labels
        return img, labels_out, img_origin


    def __initialize_category(self):
        # TODO: modify this according to specific task
        self.category_name_index_dict = {
            'person': 0
        }

    def __collect_images(self, image_root):
        """
        collect all images under image_root, according to the structure of PANDA dataset
        :param image_root:
        :return:
        """
        if not os.path.exists(image_root):
            msg = f'[PANDA] Invalid image root path: {image_root}'
            DEFAULT_LOGGER.error(msg)
            raise ValueError(msg)

        scene_name_list = os.listdir(image_root)
        image_list = []
        for scene_name in scene_name_list:
            scene_path = os.path.join(image_root, scene_name)
            file_name_list = os.listdir(scene_path)
            for file_name in file_name_list:
                if os.path.splitext(file_name)[-1].lower() not in const.VALID_IMAGE_FORMAT_LIST:
                    continue
                image_name = self.compose_image_name(scene_name, file_name)
                image_path = os.path.join(image_root, image_name)
                image_list.append(PandaImage(image_path, image_name=image_name))

        self.image_list = image_list

    @staticmethod
    def compose_image_name(scene_name, file_name):
        return scene_name + "/" + file_name

    def __collect_labels(self, annotation_file_path, target_key=VISIBLE_BODY_KEY):
        """
        read annotation info from annotation_file
        :param annotation_file_path:
        :return:
        """
        if not os.path.exists(annotation_file_path):
            msg = f'[PANDA] Invalid annotation file path: {annotation_file_path}'
            DEFAULT_LOGGER.error(msg)
            raise ValueError(msg)

        annotation_dict = {}
        with open(annotation_file_path, 'r') as annotation_file:
            annotation_dict = json.load(annotation_file)

        for panda_image in self.image_list:
            image_name = panda_image.image_name
            if image_name not in annotation_dict:
                DEFAULT_LOGGER.warning(f'[PANDA] annotation missing for image: {image_name}')
                continue
            image_annotation = annotation_dict[image_name]
            # assign image attribute
            panda_image.image_id = image_annotation['image id']
            panda_image.image_width = image_annotation['image size']['width']
            panda_image.image_height = image_annotation['image size']['height']
            panda_image.objects_list = image_annotation['objects list']
            panda_image.label_list = convert_to_label_ltrb_list(panda_image.objects_list, self.category_name_index_dict, target_key=target_key)


    @staticmethod
    def collate_fn(batch):
        img, labels, img_origin = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        img0, labels0, img0_origin = torch.stack(img, 0), torch.cat(labels, 0), torch.stack(img_origin, 0)
        return img0, labels0, img0_origin