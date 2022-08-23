from copy import deepcopy

import torch

from gigadetect.utils import geometry


def test_clip_ltrb():
    bbox = torch.tensor([[-1, 10, 50, 100]])
    target_shape = [80, 80]
    bbox_origin = deepcopy(bbox)
    geometry.clip_ltrb(bbox, target_shape)
    assert len(bbox) == len(bbox_origin)
    assert bbox[0][0] == 0
    assert bbox[0][1] == 10
    assert bbox[0][2] == 50
    assert bbox[0][3] == 80
