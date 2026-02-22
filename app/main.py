# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import pprint
import socket
import sys
from pathlib import Path

import yaml

from app.scaffold import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)


def _find_free_port():
    """Find a free TCP port by binding to port 0 and reading the OS-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def process_main(rank, fname, world_size, devices, port):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    # Set distributed env vars so train.py's init_distributed() picks them up.
    # This is the ONLY place rank/world_size/port are injected — train.py reads them.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config — init_distributed is called inside train.main()
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debugmode:
        process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"], port=0)
    else:
        num_gpus = len(args.devices)
        port = _find_free_port()
        mp.set_start_method("spawn")
        procs = []
        for rank in range(num_gpus):
            proc = mp.Process(target=process_main, args=(rank, args.fname, num_gpus, args.devices, port))
            proc.start()
            procs.append(proc)

        exit_code = 0
        failed_pid = None
        for proc in procs:
            proc.join()
            if proc.exitcode not in (0, None):
                exit_code = proc.exitcode if proc.exitcode is not None else 1
                failed_pid = proc.pid
                break

        if failed_pid is not None:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join(timeout=5)
            raise SystemExit(
                f"Distributed training worker exited with a non-zero code (pid={failed_pid}, exit_code={exit_code})."
            )

        sys.exit(0)
