#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/12 10:38
# @File     : inference.py

"""
from typing import List

import cv2
import time
import torch
import numpy as np

from gigadetect.schema.giga_detection import GigaDetection
from gigadetect.core import post_process, pre_process
from gigadetect.utils import misc, geometry
from gigadetect.models import model_loader
from gigadetect.utils.logger import DEFAULT_LOGGER


class DecDetService:
    model_path: str
    img_size: int
    conf_threshold: float
    iou_threshold: float
    device: str

    def __init__(self, model_path, img_size=640, conf_threshold=0.4, iou_threshold=0.5, device=''):
        self.__set_hyper_parameters(img_size, conf_threshold, iou_threshold, device)
        self.device = misc.get_target_device(self.device)
        self._load_pretrained_model(model_path)

    def __set_hyper_parameters(self, img_size, conf_threshold, iou_threshold, device):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.augment = False
        self.classes = False
        self.agnostic_nms = False

    def _load_pretrained_model(self, model_path):
        self.model = model_loader.load_yolov5_model(model_path, self.device, num_classes=2)
        self.model.float().fuse().eval()

    def detect_image(self, input_image) -> List[GigaDetection]:
        img = self._pre_process_image(input_image)
        preds = self.model(img, augment=self.augment)[0]

        # Apply NMS
        preds_after_nms = post_process.non_max_suppression(preds, self.conf_threshold, self.iou_threshold,
                                                           classes=self.classes,
                                                           agnostic=self.agnostic_nms)

        detections = preds_after_nms[0]
        giga_detections = []
        if detections is not None and len(detections):
            detections[:, :4] = geometry.scale_coords(img.shape[2:], detections[:, :4], input_image.shape).round()
            for det in detections:
                *xyxy, conf, cls = det
                lrtb = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                confidence = conf.item()
                category = cls.item()
                giga_detection = GigaDetection(lrtb[0], lrtb[1], lrtb[2], lrtb[3], category, confidence)
                giga_detections.append(giga_detection)
        return giga_detections

    def detect_images(self, images) -> List[List[GigaDetection]]:
        if not images:
            return []
        # set batch shape, or the image_size in one batch could be different, causing error
        batch_shape = self._get_batch_shape(images)
        # batch_shape = None

        with torch.no_grad():
            t0 = time.time()
            imgs = [self._pre_process_image(image, batch_shape) for image in images]
            imgs = torch.cat(imgs, 0)
            preds = self.model(imgs, augment=self.augment)[0]


            t1 = misc.get_synchronized_time()
            DEFAULT_LOGGER.info(f'[GigaDetect] inference cost for giga detect_images after forward: {t1- t0}')

            # Apply NMS
            preds = preds.to('cpu')
            preds_after_nms = post_process.non_max_suppression(preds, self.conf_threshold, self.iou_threshold,
                                                               classes=self.classes,
                                                               agnostic=self.agnostic_nms)
            t1 = time.time()
            DEFAULT_LOGGER.info(f'[GigaDetect] inference cost for giga detect_images after apply nms: {t1- t0}')

        giga_detections_list = []
        for (i, detections) in enumerate(preds_after_nms):
            detections = preds_after_nms[i]
            giga_detections = []
            if detections is not None and len(detections):
                detections[:, :4] = geometry.scale_coords(imgs[i].shape[1:], detections[:, :4], images[i].shape).round()
                for det in detections:
                    *xyxy, conf, cls = det
                    lrtb = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    confidence = conf.item()
                    category = cls.item()
                    giga_detection = GigaDetection(lrtb[0], lrtb[1], lrtb[2], lrtb[3], category, confidence)
                    giga_detections.append(giga_detection)
            giga_detections_list.append(giga_detections)

        t2 = time.time()
        DEFAULT_LOGGER.info(f'[GigaDetect] inference cost after generating list: {t2 - t0}')
        return giga_detections_list

    def detect_images_without_encapsulate(self, images):
        if not images:
            return []
        # set batch shape, or the image_size in one batch could be different, causing error
        batch_shape = self._get_batch_shape(images)
        # batch_shape = None

        with torch.no_grad():
            t0 = time.time()
            imgs = [self._pre_process_image(image, batch_shape) for image in images]
            DEFAULT_LOGGER.info(f'[GigaDetect] inference cost for _pre_process_image: {time.time() - t0}')

            imgs = torch.cat(imgs, 0)
            preds = self.model(imgs, augment=self.augment)[0]
            torch.cuda.synchronize()

            t1 = time.time()
            DEFAULT_LOGGER.info(f'[GigaDetect] inference cost for giga detect_images after forward: {t1- t0}')

            # Apply NMS
            preds = preds.to('cpu')
            preds_after_nms = post_process.non_max_suppression(preds, self.conf_threshold, self.iou_threshold,
                                                               classes=self.classes,
                                                               agnostic=self.agnostic_nms)
            t1 = time.time()
            DEFAULT_LOGGER.info(f'[GigaDetect] inference cost for giga detect_images after apply nms: {t1- t0}')

        giga_detections_list = []
        for (i, detections) in enumerate(preds_after_nms):
            detections = preds_after_nms[i]
            giga_detections = []
            if detections is not None and len(detections):
                detections[:, :4] = geometry.scale_coords(imgs[i].shape[1:], detections[:, :4], images[i].shape).round()
                # for det in detections:
                #     *xyxy, conf, cls = det
                #     lrtb = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                #     confidence = conf.item()
                #     category = cls.item()
                #     giga_detection = GigaDetection(lrtb[0], lrtb[1], lrtb[2], lrtb[3], category, confidence)
                #     giga_detections.append(giga_detection)
                giga_detections_list.append(detections)
            else:
                giga_detections_list.append(torch.zeros(0, 6))

        t2 = time.time()
        DEFAULT_LOGGER.info(f'[GigaDetect] inference cost after generating list: {t2 - t0}')
        return giga_detections_list

    def detect_image_file(self, file_path) -> List[GigaDetection]:
        img = cv2.imread(file_path)
        if img is None:
            return []
        return self.detect_image(img)

    def _pre_process_image(self, image, new_shape=None):
        if new_shape is None:
            new_shape = self.img_size
        img = pre_process.add_letterbox_for_image(image, new_shape=new_shape, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # from {0 - 255} to {0.0 - 1.0}
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def _get_batch_shape(self, images):

        shapes = [[img.shape[1], img.shape[0]] for img in images]
        shapes = np.array(shapes, dtype=np.float64)
        # do not execute sort operation
        aspect_ratio = shapes[:, 1] / shapes[:, 0]  # aspect ratio
        # default shapes
        shapes = [1, 1]
        mini, maxi = aspect_ratio.min(), aspect_ratio.max()
        if maxi < 1:
            shapes = [maxi, 1]
        elif mini > 1:
            shapes = [1, 1 / mini]

        stride = 32
        pad = 0

        shapes = np.ceil(np.array(shapes) * self.img_size / stride + pad).astype(
            np.int) * stride
        return shapes