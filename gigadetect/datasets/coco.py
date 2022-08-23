import os
import random
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from gigadetect.core import pre_process
from gigadetect.datasets import augment
from gigadetect.utils import file_util, coordinate
from gigadetect.config import const

from PIL import Image, ExifTags

# Get orientation exif tag
from gigadetect.utils.logger import DEFAULT_LOGGER

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s




class CocoImageAndLabelDataset(Dataset):
    path: str  # input path for current dataset
    batch_size: int
    img_sie: int

    def __init__(self, cfg, path, img_size=640, batch_size=16, augment=False, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0):

        self.cfg = cfg
        self.path = path

        # fetch images paths
        self.__fetch_image_paths(self.path)

        # fetch labels paths
        self.__fetch_label_paths(self.path)

        # set parameter
        self.batch_size_index_map = np.floor(np.arange(len(self.img_files)) / batch_size).astype(np.int)
        self.batch_count = self.batch_size_index_map[-1] + 1  # number of batches
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.single_class = single_cls
        self.augment = augment
        self.pad = pad

        # fetch shapes
        self.__fetch_shapes()

        # codes about rectangular training
        if self.rect:
            self.__set_batch_shapes()

        # Cache labels
        self.__fetch_labels()

        # initialize cached images, all set to None
        self.__initialize_images()

        # TODO: add codes about cache_images

        # TODO: add codes about Detect corrupted images

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # TODO: add logic of image_weights

        # Load image
        if self.mosaic:
            # Load mosaic
            img, current_labels = self.__load_mosaic(index)
            shapes = None
        else:
            # Load single image
            img, (h0, w0), (h, w) = self.__load_image(index)
            # Letterbox
            shape = self.batch_shapes[self.batch_size_index_map[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = pre_process.add_letterbox_for_image(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            current_labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                current_labels = x.copy()
                current_labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                current_labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                current_labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                current_labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        # Execute augment
        if self.augment:
            if not self.mosaic:
                img, labels = augment.random_affine(img, current_labels,
                                                    degrees=self.cfg.INPUT.ROTATE_DEGREES,
                                                    translate=self.cfg.INPUT.TRANSLATE,
                                                    scale=self.cfg.INPUT.SCALE,
                                                    shear=self.cfg.INPUT.SHEAR)

            augment.augment_hsv(img, hgain=self.cfg.INPUT.HSV_H, sgain=self.cfg.INPUT.HSV_S, vgain=self.cfg.INPUT.HSV_V)

            # TODO: maybe cutouts

        number_labels = len(current_labels)
        if number_labels:
            # convert xyxy to xywh
            current_labels[:, 1:5] = coordinate.ltrb2xywh(current_labels[:, 1:5])
            # Normalize coordinates 0 - 1
            current_labels[:, [2, 4]] /= img.shape[0]  # height
            current_labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            img, current_labels = self.__execute_horizontal_flip(img, current_labels)
            # random up-down flip
            img, current_labels = self.__execute_vertical_flip(img, current_labels)

        labels_out = torch.zeros((number_labels, 6))
        if number_labels:
            labels_out[:, 1:] = torch.from_numpy(current_labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def __fetch_image_paths(self, path):
        self.img_files = file_util.fetch_image_path_list(path, const.VALID_IMAGE_FORMAT_LIST)
        assert len(self.img_files) > 0, 'No images found in %s' % path

    def __fetch_label_paths(self, path):
        # labels files are supposed to be put on specific path
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]
        assert len(self.label_files) > 0, 'No labels found in %s' % path

    def __fetch_shapes(self):
        """
        In this method, cache shapes and labels, but only shapes is useful
        In fact, the `self.labels` here is invalid, because it will be override soon.
        :return:
        """
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache_dict = torch.load(cache_path)  # load
            if cache_dict['hash'] != file_util.get_hash(self.label_files + self.img_files):  # dataset changed
                cache_dict = self.__create_shape_label_cache(cache_path)  # re-cache
        else:
            cache_dict = self.__create_shape_label_cache(cache_path)  # cache
        # Read image shapes (wh)
        labels, shapes = zip(*[cache_dict[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        # TODO: remove this redundant codes
        self.labels = list(labels)

    def __create_shape_label_cache(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        cache_dict = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                # check image
                image = Image.open(img)
                image.verify()  # PIL verify
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # read labels
                l = []
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                cache_dict[img] = [l, shape]
            except Exception as e:
                cache_dict[img] = [None, None]
                DEFAULT_LOGGER.warning('Error happen in caching image and label: %s: %s' % (img, e))

        cache_dict['hash'] = file_util.get_hash(self.label_files + self.img_files)
        torch.save(cache_dict, path)
        return cache_dict

    def __set_batch_shapes(self):
        # Sort by aspect ratio
        s = self.shapes  # wh
        aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio
        rect_indexes = aspect_ratio.argsort()
        self.img_files = [self.img_files[i] for i in rect_indexes]
        self.label_files = [self.label_files[i] for i in rect_indexes]
        self.shapes = s[rect_indexes]  # wh
        aspect_ratio = aspect_ratio[rect_indexes]

        # Set training image shapes
        shapes = [[1, 1]] * self.batch_count
        for i in range(self.batch_count):
            aspect_ratio_indexs = aspect_ratio[self.batch_size_index_map == i]
            mini, maxi = aspect_ratio_indexs.min(), aspect_ratio_indexs.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
            np.int) * self.stride

    def __fetch_labels(self):
        number_of_images = len(self.label_files)
        cache_file_path = self.__label_cache_file_path()
        if os.path.isfile(cache_file_path):
            labels_data = np.load(cache_file_path, allow_pickle=True)
            if len(labels_data) == number_of_images:
                DEFAULT_LOGGER.info(f"Successfully load labels data from cache file {cache_file_path}")
                self.labels = labels_data
            else:
                self.labels = self.__create_labels_data(cache_file_path)
        else:
            self.labels = self.__create_labels_data(cache_file_path)

        # for single class case
        if self.single_class:
            for i, label in enumerate(self.labels):
                if self.labels[i].shape[0]:
                    self.labels[i][:, 0] = 0  # force dataset into single-class mode

    def __create_labels_data(self, cache_file_path=''):
        """
        This code is duplicated with `__create_shape_label_cache` logic
        Just from yolov5 project, remains to be refactor.
        :param cache_file_path:
        :return:
        """
        DEFAULT_LOGGER.info(f"No valid cached label file, start to read labels from each annotation files.")
        # initialize `self.labels` array with empty
        number_of_images = len(self.label_files)
        labels_data = [np.zeros((1, 5), dtype=np.float32)] * number_of_images

        number_of_missing, number_of_found, number_of_empty, number_of_subset, number_of_duplicate = 0, 0, 0, 0, 0
        progress_bar = tqdm(self.label_files)
        for i, file in enumerate(progress_bar):
            try:
                with open(file, 'r') as f:
                    current_label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            except:
                number_of_missing += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                continue

            if current_label.shape[0]:
                assert current_label.shape[1] == 5, '> 5 label columns: %s' % file
                assert (current_label >= 0).all(), 'negative labels: %s' % file
                assert (current_label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(current_label, axis=0).shape[0] < current_label.shape[0]:  # duplicate rows
                    number_of_duplicate += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                labels_data[i] = current_label
                number_of_found += 1  # file found
            else:
                number_of_empty += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty

            progress_bar.desc = 'Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_file_path, number_of_found, number_of_missing, number_of_empty, number_of_duplicate, number_of_images)

        assert number_of_found > 0, 'No labels found'
        DEFAULT_LOGGER.info('Saving labels data to %s for faster future loading.' % cache_file_path)
        np.save(cache_file_path, labels_data)
        return labels_data

    def __initialize_images(self):
        self.imgs = [None] * len(self.img_files)

    def __label_cache_file_path(self):
        assert self.label_files and len(self.label_files) > 0, f'Fail to fetch cached_labels_path'
        np_labels_path = str(Path(self.label_files[0]).parent) + '.npy'
        return np_labels_path

    def __load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            original_height, original_width = img.shape[:2]  # orig hw
            r = self.img_size / max(original_height, original_width)  # resize image to img_size
            if r != 1:
                # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(original_width * r), int(original_height * r)), interpolation=interp)
            return img, (original_height, original_width), img.shape[:2]  # img, hw_original, hw_resized
        else:
            # load from cache
            return self.imgs[index], self.img_original_hw[index], self.img_resized_hw[index]

    def __load_mosaic(self, index):
        """
        load a mosaic synthetic image
        :param index: 
        :return: 
        """
        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in
                             range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.__load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

            # Replicate
            # img4, labels4 = replicate(img4, labels4)

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = augment.random_affine(img4, labels4,
                                              degrees=self.cfg.INPUT.ROTATE_DEGREES,
                                              translate=self.cfg.INPUT.TRANSLATE,
                                              scale=self.cfg.INPUT.SCALE,
                                              shear=self.cfg.INPUT.SHEAR,
                                              border=self.mosaic_border)  # border to remove

        return img4, labels4

    def __execute_horizontal_flip(self, img, current_labels):
        # random left-right flip
        lr_flip = True
        if lr_flip and random.random() < 0.5:
            img = np.fliplr(img)
            if len(current_labels):
                current_labels[:, 1] = 1 - current_labels[:, 1]
        return img, current_labels

    def __execute_vertical_flip(self, img, current_labels):
        ud_flip = False
        if ud_flip and random.random() < 0.5:
            img = np.flipud(img)
            if len(current_labels):
                current_labels[:, 2] = 1 - current_labels[:, 2]
        return img, current_labels

    @staticmethod
    def collate_fn(batch):
        img, labels, path, shapes = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(labels, 0), path, shapes


def create_data_loader(cfg, path, image_size, batch_size, stride, single_cls,
                       augment=False,
                       cache=False, pad=0.0, rect=False):
    dataset = CocoImageAndLabelDataset(cfg, path, image_size, batch_size,
                                       augment=augment,  # augment images
                                       rect=rect,  # rectangular training
                                       cache_images=cache,
                                       single_cls=single_cls,
                                       stride=int(stride),
                                       pad=pad)

    batch_size = min(batch_size, len(dataset))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=CocoImageAndLabelDataset.collate_fn)
    return data_loader, dataset
