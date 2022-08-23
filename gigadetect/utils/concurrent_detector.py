#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/10/15 18:56
# @File     : concurrent.py.py

"""
import argparse

import cv2
import os

import time

from gigadetect.core.decdet_inference import DecDetService
from gigadetect.utils import visualizer
from gigadetect.utils.logger import DEFAULT_LOGGER
from concurrent.futures import ThreadPoolExecutor

def detect_branch(giga_detect_service, images):
    t0 = time.time()
    giga_detections = giga_detect_service.detect_images(images)
    t1 = time.time()
    print(f'inference cost {t1 - t0}s.')
    return giga_detections

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--image_path', type=str, default='', help='the path of input image')
    parser.add_argument('--show-image', action='store_true',
                        help='display the detection results, do not use it if in command line env')
    parser.add_argument('--save_path', type=str, default=None,
                        help='the file path to store image with detection result')

    args = parser.parse_args()
    return args


def post_process(args, image, giga_detections):
    display_image = args.show_image
    if display_image:
        visualizer.show_detection_result(image, [], giga_detections, target_size=1080)

    save_path = args.save_path
    if save_path:
        image_with_detection = visualizer.draw_detection_result_to_image(image, giga_detections)
        cv2.imwrite(save_path, image)


if __name__ == "__main__":

    args = parse_args()

    # load input image
    image_path = args.image_path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Error occurs when loading input image: `{image_path}`')

    # load pre-trained model
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise ValueError(f'Error occurs when loading pre-trained model: `{model_path}`')
    giga_detect_service = DecDetService(model_path, conf_threshold=0.1)
    giga_detect_service_2 = DecDetService(model_path, conf_threshold=0.1)

    thread_pool_size = 8
    giga_detect_service_pool = [DecDetService(model_path, conf_threshold=0.1) for _ in range(thread_pool_size)]

    """
    # execute inference
    t0 = time.time()
    giga_detections = giga_detect_service.detect_image(image)
    t1 = time.time()
    print(f'inference cost {t1 - t0}s.')

    # print detections info
    [print(giga_detection) for giga_detection in giga_detections]
    
    # [optional] display/save annotated image
    post_process(args, image, giga_detections)
    """

    images = [image for _ in range(0, 16)]

    # serially
    t21 = time.time()
    result = [detect_branch(giga_detect_service, images) for giga_detect_service in giga_detect_service_pool]
    t22 = time.time()
    DEFAULT_LOGGER.info(f'#serially# cost time: {t22-t21} s')
    print(len(result))


    # concurrently
    t31 = time.time()

    with ThreadPoolExecutor(thread_pool_size) as executor:
        args = [[_, images] for _ in giga_detect_service_pool]
        result = executor.map(lambda p: detect_branch(*p), args)
    t32 = time.time()
    DEFAULT_LOGGER.info(f'#concurrently# cost time: {t32-t31} s')
    print(len(list(result)))




    DEFAULT_LOGGER.info("The inference procedure is finished.")
