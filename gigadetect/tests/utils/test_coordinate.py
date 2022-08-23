import numpy as np

from gigadetect.utils import coordinate


def test_xywh2ltrb():
    xywh = np.array([[100, 100, 100, 100]])
    ltrb = coordinate.xywh2ltrb(xywh)
    assert len(ltrb) == len(xywh)
    assert ltrb[0][0] == 50
    assert ltrb[0][1] == 50
    assert ltrb[0][2] == 150
    assert ltrb[0][3] == 150
    # TODO: add more test cases


def test_ltrb2xywh():
    ltrb = np.array([[50, 50, 150, 150]])
    xywh = coordinate.ltrb2xywh(ltrb)
    assert len(xywh) == len(ltrb)
    assert xywh[0][0] == 100
    assert xywh[0][1] == 100
    assert xywh[0][2] == 100
    assert xywh[0][3] == 100
    # TODO: add more test cases
