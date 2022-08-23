#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/1/30 15:45
# @File     : parser.py

"""
import argparse

import os

from gigadetect.config.cfg_node import get_decdet_default_cfg, get_gigadet_default_cfg
from gigadetect.utils.logger import DEFAULT_LOGGER, setup_logger


def default_argument_parser():
    parser = argparse.ArgumentParser(description="GigaDet")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_cfg(args, merge_from_list=True, prepare_env=True):
    cfg = get_decdet_default_cfg()
    cfg.merge_from_file(args.config_file)
    if merge_from_list:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    if prepare_env:
        prepare_current_env(cfg, args)
    return cfg

def setup_cfg_for_gigadet_eval(args, prepare_env=True):
    cfg = get_gigadet_default_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if prepare_env:
        prepare_current_env(cfg, args)
    return cfg

def prepare_current_env(cfg, args):

    # create dir to save config, logs, and models.
    output_dir = cfg.OUTPUT.ROOT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # setup logger
    setup_logger(cfg, output_dir)

    if output_dir:
        # save config
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        DEFAULT_LOGGER.info("Full config saved to {}".format(os.path.abspath(path)))


    # print config info
    if hasattr(args, "config_file") and args.config_file != "":
        DEFAULT_LOGGER.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, open(args.config_file, "r").read()
            )
        )
    DEFAULT_LOGGER.info("Running with full config:\n{}".format(cfg))

