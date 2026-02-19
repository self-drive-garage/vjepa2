# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


def _read_int_env(name):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    if (rank is None) or (world_size is None):
        # torchrun-style environment variables (multi-node and single-node).
        torchrun_world_size = _read_int_env("WORLD_SIZE")
        torchrun_rank = _read_int_env("RANK")
        if (torchrun_world_size is not None) and (torchrun_rank is not None):
            world_size = torchrun_world_size
            rank = torchrun_rank
        else:
            try:
                world_size = int(os.environ["SLURM_NTASKS"])
                rank = int(os.environ["SLURM_PROCID"])
            except Exception:
                logger.info("Distributed env vars not set (distributed training not available)")
                world_size, rank = 1, 0
                return world_size, rank

    if "MASTER_ADDR" not in os.environ:
        if "HOSTNAME" in os.environ:
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        else:
            os.environ["MASTER_ADDR"] = "localhost"

    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl").lower()
    try:
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available ({backend=}): {e}")

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
