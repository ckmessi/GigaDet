import os

import cv2
import torch
from tqdm import tqdm

from gigadetect.core import evaluater, pre_process, post_process
from gigadetect.core.evaluater import EvaluateStatistics
from gigadetect.core.decdet_inference import DecDetService
from gigadetect.datasets.panda import PandaImageAndLabelDataset
from gigadetect.schema.giga_detection import GigaDetection, GigaPatch
from gigadetect.utils import visualizer, misc, coordinate


def crop_image(img, left, top, right, bottom):
    h, w = img.shape[0], img.shape[1]

    left = int(w * left)
    right = int(w * right)
    top = int(h * top)
    bottom = int(h * bottom)

    img_cropped = img[top:bottom, left:right]
    return img_cropped


def crop_region_contains_patch(img, left, top, right, bottom):
    h, w = img.shape[0], img.shape[1]


    # test1: crop twice from center
    print(f'Patch size is　w={w*(right-left)}, h={h*(bottom-top)}')
    bbox_w = max(right - left, bottom - top)
    left_pos = int(w * ((left + right) / 2 - bbox_w))
    left_pos = max(left_pos, 0)
    right_pos = int(w * ((left + right) / 2 + bbox_w))
    right_pos = min(right_pos, w)
    top_pos = int(h * ((top + bottom) / 2 - bbox_w))
    top_pos = max(top_pos, 0)
    bottom_pos = int(h * ((top + bottom) / 2 + bbox_w))
    bottom_pos = min(bottom_pos, h)

    # test2: use fixed size
    left_pos = int(w * ((left + right) / 2) + -1000)
    left_pos = max(left_pos, 0)
    right_pos = int(w * ((left + right) / 2) + 1000)
    right_pos = min(right_pos, w)
    top_pos = int(h * ((top + bottom) / 2) - 1000)
    top_pos = max(top_pos, 0)
    bottom_pos = int(h * ((top + bottom) / 2) + 1000)
    bottom_pos = min(bottom_pos, h)

    img_cropped = img[top_pos:bottom_pos, left_pos:right_pos]
    img_h = bottom_pos - top_pos
    img_w = right_pos - left_pos

    giga_patch = GigaPatch(img_cropped, h, w, img_h, img_w, left_pos, top_pos)
    return giga_patch


if __name__ == "__main__":

    image_root = 'D:/Data/Dataset/PANDA_DATASET/PANDA/PANDA_IMAGE/image_train/'
    # image_root = 'D:/tmp/Image'
    annotation_file_path = 'D:/Data/Dataset/PANDA_DATASET/PANDA/PANDA_IMAGE/image_annos/person_bbox_train.json'
    model_path = 'D:/Data/model/yolov5/yolov5m_save_without_training_41_state_dict.pth'
    dataset = PandaImageAndLabelDataset(image_root, annotation_file_path)

    batch_size = 2
    batch_size = min(batch_size, len(dataset))
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    num_workers = 0
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=PandaImageAndLabelDataset.collate_fn
                                              )

    giga_detect_service = DecDetService(model_path)

    num_batches = len(data_loader)
    for i, (imgs, labels, imgs_origin) in enumerate(data_loader):
        # labels format is ltrb
        imgs = imgs.numpy()
        imgs_origin = imgs_origin.numpy()
        print(f"iterator index #{i}")
        print(imgs.shape, labels.shape)

        global_detections = []
        for index_in_batch in range(0, imgs.shape[0]):
            cur_labels = labels[labels[:, 0] == index_in_batch]
            h, w = imgs[index_in_batch].shape[0], imgs[index_in_batch].shape[1]
            giga_detections = [GigaDetection(label[2] * w, label[3] * h, label[4] * w, label[5] * h, label[1], 1.0) for
                               label in cur_labels]
            #visualizer.show_detection_result(imgs[index_in_batch], giga_detections, [], target_size=1080, draw_label=False)

            # display a patch
            global_detection_in_batch_sample = []
            if len(cur_labels):
                for label_index in range(0, len(cur_labels)):
                    label = cur_labels[label_index]
                    giga_patch = crop_region_contains_patch(imgs_origin[index_in_batch], label[2], label[3], label[4],
                                                            label[5])
                    img_cropped = giga_patch.data
                    # execute detect once
                    giga_detections = giga_detect_service.detect_image(img_cropped)
                    current_global_detections = giga_patch.convert_to_global_detection_absolute_value(giga_detections)
                    global_detection_in_batch_sample.extend(current_global_detections)
                    # visualizer.show_detection_result(img_cropped, [], giga_detections, target_size=1280)
                    # cv2.imshow('patch', img_cropped)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    print(f'finish detection in patch {label_index}，total {len(giga_detections)} object detected')
            global_detections.append(global_detection_in_batch_sample)

        device = misc.get_target_device('cpu')
        # calculate metrics for current image
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        output = [
            torch.FloatTensor([gd.convert_to_ltrb_confidence_category() for gd in global_detection_in_batch_sample]) for
            global_detection_in_batch_sample in global_detections
        ]

        output = post_process.non_max_suppression_for_batch_outputs(output, conf_thres=0.001, iou_thres=0.6, merge=False)
        # targets should be xywh format
        # label_list_ltrb = np.array(label_list_ltrb)
        # label_list_xywh = label_list_ltrb
        # label_list_xywh[:, 1:5] = coordinate.ltrb2xywh(label_list_ltrb[:, 1:5])


        targets_ltrb = torch.tensor(labels)
        targets_xywh = targets_ltrb
        # labels shape is batch_size x 6，the 6 items represents `batch_index`, `category`, `l`, `t`, `r`, `b`
        targets_xywh[:, 2:6] = coordinate.ltrb2xywh(targets_ltrb[:, 2:6])
        nb, origin_height, origin_width, _ = imgs_origin.shape  # batch size, channels, height, width
        whwh = torch.Tensor([origin_width, origin_height, origin_width, origin_height]).to(device)
        epoch_stats, seen_object_count = evaluater.calculate_metric(output, targets_xywh, niou, iouv, origin_height,
                                                                    origin_width, device, whwh)

        evaluate_statistics = EvaluateStatistics()
        evaluate_statistics.image_count = seen_object_count
        evaluate_statistics.compute_statistics(epoch_stats, 1)
        evaluate_statistics.print_title()
        evaluate_statistics.print_results(epoch_stats, "person", True, 0, 0, 640, batch_size, True)

    print("Hello")
