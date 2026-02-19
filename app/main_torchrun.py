# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pprint
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    help="name of config file to load",
    default="configs.yaml",
)


def main():
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info("called-params %s", args.fname)

    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    logger.info("loaded params...")

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = Path(params["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        params_path = folder / "params-pretrain.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    logger.info("Running... (rank: %d/%d, local_rank=%d)", rank, world_size, local_rank)
    app_main(params["app"], args=params)


if __name__ == "__main__":
    main()
