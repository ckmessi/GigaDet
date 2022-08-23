#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/12 11:06
# @File     : visualizer.py

"""
from typing import List

import cv2

from gigadetect.schema.giga_detection import GigaDetection

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (225, 255, 255)


def draw_detection_info(image, detection: GigaDetection, color=COLOR_GREEN, thickness=None, draw_label=True):
    bbox = detection.bbox
    p1 = (int(bbox.left), int(bbox.top))
    p2 = (int(bbox.right), int(bbox.bottom))
    thickness = thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    cv2.rectangle(image, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    if draw_label:
        score = '%.2f' % detection.confidence
        label = '%g' % detection.category
        text = score + " " + label
        font_thickness = max(thickness - 1, 1)
        text_size = cv2.getTextSize(text, 0, fontScale=thickness / 3, thickness=font_thickness)[0]
        p2 = p1[0] + text_size[0], p1[1] - text_size[1] - 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        p3 = (p1[0], p1[1] - 2)
        cv2.putText(image, text, p3, 0, thickness / 3, COLOR_WHITE, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image


def draw_detection_result_to_image(image, detections: List[GigaDetection], color=COLOR_GREEN, draw_label=True):
    for detection in detections:
        image = draw_detection_info(image, detection, color, draw_label=draw_label)
    return image


def show_detection_result(image, gt_detection_list: List[GigaDetection], predict_detection_list: List[GigaDetection], target_size=None, draw_label=True):
    # draw gt bounding boxes using green rectangle
    image = draw_detection_result_to_image(image, gt_detection_list, COLOR_GREEN, draw_label=draw_label)
    # draw predict bounding boxes using red rectangle
    image = draw_detection_result_to_image(image, predict_detection_list, (255, 0, 0), draw_label=draw_label)

    if target_size:
        ratio = target_size / max(image.shape[0], image.shape[1])
        interp = cv2.INTER_AREA
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=interp)
    cv2.namedWindow('detection_result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('detection_result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
