#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/11 13:13
# @File     : logger.py

"""


import sys

import os
from loguru import logger

DEFAULT_LOGGER = logger
GIGA_DETECT_LOGGER = logger

__all__ = [
    'DEFAULT_LOGGER',
    'GIGA_DETECT_LOGGER'
]


def setup_logger(cfg, output_dir):

    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    #  TODO: load log_level from config
    log_level = cfg.INFO.LOG_LEVEL
    DEFAULT_LOGGER.remove()
    DEFAULT_LOGGER.add(sys.stderr, format=LOG_FORMAT, level=log_level)

    # meanwhile set to file
    if output_dir:
        log_file_path = os.path.join(output_dir, "log.txt")
        DEFAULT_LOGGER.add(log_file_path, format=LOG_FORMAT, level=log_level)

    # TODO: how to deal with different logger

    DEFAULT_LOGGER.info(f"** Here starts new block\n\n\n\n\n")
    DEFAULT_LOGGER.info(f"** Logger setup complete.")
