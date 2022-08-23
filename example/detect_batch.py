#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/2 16:13
# @File     : detect_batch.py

"""

import argparse

import cv2
import os

import time

from gigadetect.core.decdet_inference import DecDetService
from gigadetect.utils import visualizer
from gigadetect.utils.logger import DEFAULT_LOGGER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--input_dir', type=str, default='', help='the path of input image dir')
    parser.add_argument('--show-image', action='store_true',
                        help='display the detection results, do not use it if in command line env')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='the file path to store image with detection result')

    args = parser.parse_args()
    return args


def post_process(args, images, giga_detections_list, image_paths):
    display_image = args.show_image
    if display_image:
        for (i, image) in enumerate(images):
            giga_detections = giga_detections_list[i]
            visualizer.show_detection_result(image, [], giga_detections, target_size=1080)

    output_dir = args.output_dir
    if output_dir:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for (i, image) in enumerate(images):
            image_with_detection = visualizer.draw_detection_result_to_image(image, giga_detections_list[i])
            save_path = os.path.join(output_dir, os.path.basename(image_paths[i]))
            DEFAULT_LOGGER.info(f'saving image to {save_path}...')
            ret = cv2.imwrite(save_path, image_with_detection)


if __name__ == "__main__":

    args = parse_args()

    # load input image
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f'Error occurs when loading input path: `{input_dir}`')

    image_paths = [os.path.join(input_dir, image_name) for image_name in os.listdir(input_dir)]
    images = [cv2.imread(image_path) for image_path in image_paths]

    # load pre-trained model
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise ValueError(f'Error occurs when loading pre-trained model: `{model_path}`')
    giga_detect_service = DecDetService(model_path, img_size=640)

    # execute inference
    t0 = time.time()
    giga_detections_list = giga_detect_service.detect_images(images)
    t1 = time.time()
    print(f'inference cost {t1 - t0}s.')

    # print detections info
    # [[print(giga_detection) for giga_detection in giga_detections] and print("") for giga_detections in giga_detections_list]

    # [optional] display/save annotated image
    post_process(args, images, giga_detections_list, image_paths)

    DEFAULT_LOGGER.info("The inference procedure is finished.")
