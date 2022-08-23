import torch
import numpy as np


def xywh2ltrb(xywh):
    """
    Convert [n x 4] boxes from `n x [x, y, w, h]` format to `n x [left, top, right, bottom]` format
    :param xywh:
    :return:
    """
    ltrb = torch.zeros_like(xywh) if isinstance(xywh, torch.Tensor) else np.zeros_like(xywh)
    ltrb[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    ltrb[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    ltrb[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    ltrb[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return ltrb


def ltrb2xywh(ltrb):
    """
    Convert [n x 4] boxes from `n x [left, top, right, bottom]` format to `n x [x, y, w, h]` format
    :param ltrb:
    :return:
    """
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    xywh = torch.zeros_like(ltrb) if isinstance(ltrb, torch.Tensor) else np.zeros_like(ltrb)
    xywh[:, 0] = (ltrb[:, 0] + ltrb[:, 2]) / 2  # center x
    xywh[:, 1] = (ltrb[:, 1] + ltrb[:, 3]) / 2  # center y
    xywh[:, 2] = ltrb[:, 2] - ltrb[:, 0]  # width
    xywh[:, 3] = ltrb[:, 3] - ltrb[:, 1]  # height
    return xywh
