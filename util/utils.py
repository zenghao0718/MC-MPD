"""Distributed setup and reproducibility helpers for DDFSD."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_dist(args):
    """Prepare one CUDA device per distributed process."""

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.master_addr = os.environ["MASTER_ADDR"]
    args.master_port = os.environ["MASTER_PORT"]

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    if args.local_rank == 0:
        print(f"Initial process group with tcp://{args.master_addr}:{args.master_port}.")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=args.world_size,
        rank=args.rank,
    )
    if args.seed is not None:
        set_seed(args.seed + args.rank)


def set_seed(seed):
    """Fix Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
