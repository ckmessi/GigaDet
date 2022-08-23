#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/11 13:29
# @File     : dummy_model.py

"""
import torch

class DummyModel(torch.nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
        )
        self.fc = torch.nn.Linear(16 * 224 * 224, 100)

    def forward(self, x):
        out = self.layer1(x)
        out = self.fc(out)
        return out