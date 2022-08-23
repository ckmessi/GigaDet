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
    parser.add_argument('--image_path', type=str, default='', help='the path of input image')
    parser.add_argument('--show-image', action='store_true', help='display the detection results, do not use it if in command line env')
    parser.add_argument('--save_path', type=str, default=None, help='the file path to store image with detection result')

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

    # execute inference
    t0 = time.time()
    giga_detections = giga_detect_service.detect_image(image)
    t1 = time.time()
    print(f'inference cost {t1 - t0}s.')

    # print detections info
    [print(giga_detection) for giga_detection in giga_detections]


    # [optional] display/save annotated image
    post_process(args, image, giga_detections)
  

    DEFAULT_LOGGER.info("The inference procedure is finished.")
