#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/12 10:44
# @File     : giga_detection.py

"""
from typing import List

import torch

from gigadetect.utils import geometry


class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int

    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)


class BaseDetection:
    bbox: BoundingBox
    category: int
    confidence: float

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence


class GigaDetection(BaseDetection):
    pass

    def __str__(self):
        return f'[GigaDetection]: left={self.bbox.left}, top={self.bbox.top}, right={self.bbox.right}, bottom={self.bbox.bottom}'

    def convert_to_ltrb_confidence_category(self):
        return [self.bbox.left, self.bbox.top, self.bbox.right, self.bbox.bottom, self.confidence, self.category]

    @staticmethod
    def buildGigaDetectionFromModelOutputs(outputs):
        detections = outputs[0]
        giga_detections = []
        if detections is not None and len(detections):
            for det in detections:
                *xyxy, conf, cls = det
                lrtb = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                confidence = conf.item()
                category = cls.item()
                giga_detection = GigaDetection(lrtb[0], lrtb[1], lrtb[2], lrtb[3], category, confidence)
                giga_detections.append(giga_detection)
        return giga_detections

    @staticmethod
    def filterOnlyPersonCategory(giga_detections_list):
        # TODO：check this logic, only get category == 0(person) results
        TARGET_CATEGORY = 0
        filtered_giga_detections_list = [
            list(filter(lambda giga_detection: giga_detection.category == TARGET_CATEGORY, giga_detections)) for
            giga_detections in giga_detections_list
        ]
        return filtered_giga_detections_list


class GigaPatch:
    data: object
    origin_height: int
    origin_width: int
    height: int
    width: int
    left_left: int
    top_top: int
    possible_score: float

    def __init__(self, data, origin_height, origin_width, height, width, left_left, top_top, possible_score=1.0):
        self.data = data
        self.origin_height = origin_height
        self.origin_width = origin_width
        self.height = height
        self.width = width
        self.left_left = left_left
        self.top_top = top_top

        self.possible_score = possible_score

    def convert_to_global_detection_absolute_value(self, giga_detection: List[GigaDetection]):
        """
        :param giga_detection:
        :return:
        """
        return [GigaDetection(d.bbox.left + self.left_left, d.bbox.top + self.top_top, d.bbox.right + self.left_left,
                              d.bbox.bottom + self.top_top, category=d.category, confidence=d.confidence) for d in
                giga_detection]

    def convert_to_global_detection_absolute_value_without_encapsulate(self, detections):
        detections[:, 0] += self.left_left
        detections[:, 2] += self.left_left
        detections[:, 1] += self.top_top
        detections[:, 3] += self.top_top
        return detections