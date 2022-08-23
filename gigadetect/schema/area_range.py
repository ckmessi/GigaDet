#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/23 11:06
# @File     : area_range.py

"""

class AreaRange:

    SIZE_SMALL = 32 * 3
    SIZE_MEDIUM = 96 * 3
    SIZE_LARGE = 1e5

    name: str
    area_min: float
    area_max: float
    area_label: str

    def __init__(self, name: str, area_min: float, area_max: float, area_label: str):
        self.name = name
        self.area_min = area_min
        self.area_max = area_max
        self.area_label = area_label


AREA_SMALL = AreaRange('small', 0, AreaRange.SIZE_SMALL*AreaRange.SIZE_SMALL, 'small')
AREA_MEDIUM = AreaRange('medium', AreaRange.SIZE_SMALL*AreaRange.SIZE_SMALL, AreaRange.SIZE_MEDIUM*AreaRange.SIZE_MEDIUM, 'medium')
AREA_LARGE = AreaRange('large', AreaRange.SIZE_MEDIUM*AreaRange.SIZE_MEDIUM, AreaRange.SIZE_LARGE*AreaRange.SIZE_LARGE, 'large')
AREA_ALL = AreaRange('all', 0, AreaRange.SIZE_LARGE*AreaRange.SIZE_LARGE, 'all')