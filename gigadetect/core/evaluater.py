#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/13 10:47
# @File     : evaluater.py

"""
import numpy as np

import torch
from tqdm import tqdm

from gigadetect.core import losser, post_process
from gigadetect.schema.area_range import AreaRange, AREA_ALL
from gigadetect.utils import misc, coordinate, geometry
from gigadetect.utils.logger import DEFAULT_LOGGER


class EvaluateStatistics:
    image_count: int
    precision: object
    recall: object
    ap: object
    f1: object
    ap_class: object
    ap50: object
    mp: object
    mr: object
    map50: object
    map: object
    number_of_targets_per_class: object
    number_of_classes: int

    def __init__(self):
        self.stats = []
        self.image_count = 0
        self.precision = []
        self.recall = []
        self.ap = []
        self.f1 = []
        self.ap_class = []
        self.ap50 = []
        self.mp = []
        self.mr = []
        self.map50 = []
        self.map = []
        self.number_of_targets_per_class = []
        self.number_of_classes = 0

    def compute_statistics(self, stats, number_of_classes):
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = misc.ap_per_class(*stats)
            # p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            # NOTE: different from original YOLO metric, here `r` means `ar`
            p, r, ap50, ap = p[:, 0], r.mean(axis=1), ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=number_of_classes)  # number of targets per class
            self.precision = p
            self.recall = r
            self.ap = ap
            self.f1 = f1
            self.ap_class = ap_class
            self.ap50 = ap50
            self.mp = mp
            self.mr = mr
            self.map50 = map50
            self.map = map
            self.number_of_targets_per_class = nt
        else:
            nt = torch.zeros(1)
            self.number_of_targets_per_class = nt

        self.number_of_classes = number_of_classes

    def print_results(self, stats, category_names, verbose, t0, t1, image_size, batch_size, training):
        pf = '%20s' + '%12.3g' * 6  # print format
        # Print results
        self.__print_results_total(pf)
        # Print results per class
        self.__print_results_per_class(pf, verbose, stats, category_names)
        # Print speeds
        self.__print_speeds(t0, t1, image_size, batch_size, training)

    def __print_results_total(self, pf):
        desc = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        DEFAULT_LOGGER.info(desc)
        DEFAULT_LOGGER.info(pf % ('all', self.image_count, self.number_of_targets_per_class.sum(), self.mp, self.mr, self.map50, self.map))

    def __print_results_per_class(self, pf, verbose, stats, category_names):
        # Print results per class
        # TODO: use self.stats rather than augment
        if verbose and self.number_of_classes > 1 and len(stats):
            for i, c in enumerate(self.ap_class):
                DEFAULT_LOGGER.info(pf % (category_names[c], self.image_count, self.number_of_targets_per_class[c], self.precision[i], self.recall[i], self.ap50[i], self.ap[i]))

    def __print_speeds(self, t0, t1, image_size, batch_size, training):
        t = tuple(x / self.image_count * 1E3 for x in (t0, t1, t0 + t1)) + (image_size, image_size, batch_size)  # tuple
        if not training:
            DEFAULT_LOGGER.info(f'object count: {self.image_count}')
            DEFAULT_LOGGER.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


    def print_title(self):
        desc = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        DEFAULT_LOGGER.info(desc)


def calculate_metric(output, targets, niou, iouv, height, width, device, whwh, area_range: AreaRange = AREA_ALL):
    """

    :return:
    """
    seen_image_count = 0
    stats = []

    for si, pred in enumerate(output):
        labels = targets[targets[:, 0] == si, 1:]
        labels = filter_label_bbox_list(labels, height, width, area_range)
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        seen_image_count += 1

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Clip boxes to image bounds
        geometry.clip_ltrb(pred, (height, width))
        pred = filter_pred_bbox_list(pred, area_range)

        # Append to pycocotools JSON dictionary
        # TODO: add save json logic

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = coordinate.xywh2ltrb(labels[:, 1:5]) * whwh

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = geometry.box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    for j in (ious > iouv[0]).nonzero():
                        d = ti[i[j]]  # detected target
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    return stats, seen_image_count


def filter_label_bbox_list(labels, height, weight, area_range: AreaRange):
    nl = len(labels)
    if not nl:
        return labels
    if area_range == AREA_ALL:
        return labels
    filtered_labels = labels[(labels[:, 3] * weight)*(labels[:, 4] * height) >= area_range.area_min]
    filtered_labels = filtered_labels[(filtered_labels[:, 3] * weight)*(filtered_labels[:, 4] * height) < area_range.area_max]
    return filtered_labels

def filter_pred_bbox_list(preds, area_range: AreaRange):
    nl = len(preds)
    if not nl:
        return preds
    if area_range == AREA_ALL:
        return preds
    filtered_preds = preds[(preds[:, 2]-preds[:, 0])*(preds[:, 3]-preds[:, 1]) >= area_range.area_min]
    filtered_preds = filtered_preds[(filtered_preds[:, 2] - filtered_preds[:, 0])*(filtered_preds[:, 3] - filtered_preds[:, 1]) < area_range.area_max]
    return filtered_preds


def evaluate(cfg, model, data_loader, half,
             training=False,
             batch_size=16,
             imgsz=640,
             conf_thres=0.001,
             iou_thres=0.6,  # for NMS
             augment=False,
             verbose=False,
             merge=False):

    # Initialize/load model and set device
    model.eval()
    if half:
        model.half()
    device = next(model.parameters()).device

    # Configure
    num_classes = cfg.DATASETS.NUM_CLASSES
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen_img_count = 0
    category_names = cfg.DATASETS.CLASS_NAMES

    # prepare to iterate
    tqdm_description = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    t0, t1 = 0., 0.
    loss = torch.zeros(3, device=device)
    stats = []

    # start iterate
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(data_loader, desc=tqdm_description)):
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = misc.get_synchronized_time()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += misc.get_synchronized_time() - t

            # Compute loss
            if training:
                # TODO: add correct logic of compute_loss
                loss += losser.compute_loss(cfg, [x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = misc.get_synchronized_time()
            output = post_process.non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += misc.get_synchronized_time() - t

        # Statistics per image

        epoch_stats, img_count = calculate_metric(output, targets, niou, iouv, height, width, device, whwh)
        seen_img_count += img_count
        stats.extend(epoch_stats)
        # Plot images
        # TODO: plot result images

    # Compute Statistics
    evaluate_statistics = EvaluateStatistics()
    evaluate_statistics.image_count = seen_img_count
    evaluate_statistics.compute_statistics(stats, num_classes)

    # print result
    evaluate_statistics.print_results(stats, category_names, verbose, t0, t1, imgsz, batch_size, training)

    # TODO: add save json logic

    # Return results
    model.float()  # for training
    maps = np.zeros(num_classes) + evaluate_statistics.map
    for i, c in enumerate(evaluate_statistics.ap_class):
        maps[c] = evaluate_statistics.ap[i]
    return (evaluate_statistics.mp, evaluate_statistics.mr, evaluate_statistics.map50, evaluate_statistics.map, *(loss.cpu() / len(data_loader)).tolist()), maps, t

