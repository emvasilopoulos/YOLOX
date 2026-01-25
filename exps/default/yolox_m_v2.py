#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import ExpV2

"""
according to ExpV2:
- input_size is used to set both test_size and input_size
- input_size is used by internall dataset objects (e.g. COCODataset, MosaicDetection & YoloBatchSampler that use properties img_size)

All datasets like COCODataset and MosaicDetection inherit from Dataset (yolox/data/datasets/datasets_wrapper.py) that expects "input_demension" property to be set.

according to yolox/data/datasets/datasets_wrapper.py
- input_dimension (tuple): (width,height) tuple with default dimensions of the network
"""

class Exp(ExpV2):
    def __init__(self, input_width=640, input_height=640):
        super(Exp, self).__init__((input_width, input_height))
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        