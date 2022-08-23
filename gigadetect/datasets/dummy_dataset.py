#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/11 15:58
# @File     : dummy_dataset.py

"""
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __len__(self):
        return 0


    def __getitem__(self, index):
        return None