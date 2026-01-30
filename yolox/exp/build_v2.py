#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file_v2(exp_file, input_width=640, input_height=640):
    try:
        sys.path.append(os.path.dirname(exp_file))
        module_path = os.path.basename(exp_file)
        current_exp = importlib.import_module(module_path.split(".")[0])
        exp = current_exp.Exp(input_width=input_width,
                              input_height=input_height)
    except Exception:
        raise ImportError(
            "{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name_v2(exp_name, input_width=640, input_height=640):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-m": "yolox_m_v2.py",
    }
    try:
        filename = filedict[exp_name]
    except KeyError:
        raise KeyError(
            "Exp name {} not found in default experiments.".format(exp_name))
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file_v2(exp_path,
                              input_width=input_width,
                              input_height=input_height)


def get_exp_v2(exp_file, exp_name, input_width=640, input_height=640):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolox-m",
    """
    assert (exp_file is not None
            or exp_name is not None), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file_v2(exp_file,
                                  input_width=input_width,
                                  input_height=input_height)
    else:
        return get_exp_by_name_v2(exp_name,
                                  input_width=input_width,
                                  input_height=input_height)
