import argparse

from gigadetect.config import parser
from gigadetect.core import decdet_trainer
from gigadetect.utils.logger import DEFAULT_LOGGER



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    gigadet_cfg = parser.setup_cfg(args)

    # execute train
    decdet_trainer.train(gigadet_cfg)
    DEFAULT_LOGGER.info("The awesome training procedure is finished.")
