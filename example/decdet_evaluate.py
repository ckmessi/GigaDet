import argparse
import torch

from gigadetect.config import parser
from gigadetect.core import evaluater
from gigadetect.datasets.coco import create_data_loader
from gigadetect.models import model_loader
from gigadetect.utils import misc


def warm_up_forward(model, device, image_size):
    img = torch.zeros((1, 3, image_size, image_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # TODO：add options for `augment`, `merge`, `single-cls`
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gigadet_cfg = parser.setup_cfg(args)

    # load model
    batch_size = gigadet_cfg.TEST.BATCH_SIZE
    device = misc.get_target_device(gigadet_cfg.MODEL.CUDA_DEVICE, batch_size=batch_size)
    half = device.type != 'cpu' and torch.cuda.device_count() == 1  # half precision only supported on single-GPU
    model = model_loader.load_pretrained_model(gigadet_cfg, device)
    model.eval()
    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    if half:
        model.half()  # to FP16

    # run once to warm up
    image_size = misc.check_img_size(gigadet_cfg.INPUT.IMAGE_SIZE_TEST[0], s=model.stride.max())
    warm_up_forward(model, device, image_size)

    # data loader
    data_loader = create_data_loader(gigadet_cfg, gigadet_cfg.DATASETS.VAL_ROOT, image_size, batch_size, model.stride.max(), False,
                                     augment=False, cache=False, pad=0.5, rect=True)[0]

    # execute evaluate
    evaluater.evaluate(gigadet_cfg, model, data_loader, half,
                       training=False,
                       batch_size=batch_size,
                       imgsz=image_size,
                       conf_thres=gigadet_cfg.TEST.DECDET.CONF_THRESHOLD,
                       iou_thres=gigadet_cfg.TEST.DECDET.NMS_IOU_THRESHOLD,
                       augment=False,
                       verbose=gigadet_cfg.TEST.VERBOSE)

    # TODO: add logic of study task
