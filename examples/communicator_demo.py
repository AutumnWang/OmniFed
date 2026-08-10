# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal OmniFed example: use TorchDistCommunicator directly, no Ray, no dataset.

Spawns 3 local processes that broadcast a tensor from rank 0 and then
average a per-rank tensor across all ranks — the two primitives every
FL algorithm in this framework is built on.

Run from anywhere:

    python examples/communicator_demo.py
"""

import sys
from pathlib import Path

# Repo root must be on sys.path so `src.omnifed` resolves (no pip install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.multiprocessing as mp

from src.omnifed.communicator import AggregationOp, TorchDistCommunicator

WORLD_SIZE = 3
PORT = 29517  # any free local port


def worker(rank: int) -> None:
    comm = TorchDistCommunicator(
        rank=rank,
        world_size=WORLD_SIZE,
        master_addr="127.0.0.1",
        master_port=PORT,
        backend="gloo",  # CPU-only, works everywhere
    )
    comm.setup()

    # 1) Broadcast: rank 0's tensor overwrites everyone else's.
    t = torch.full((3,), float(rank))
    t = comm.broadcast(t, src=0)
    print(f"[rank {rank}] after broadcast : {t.tolist()}  (expected all 0.0)")

    # 2) Aggregate: element-wise MEAN over ranks 0,1,2 -> 1.0 everywhere.
    t = torch.full((3,), float(rank))
    t = comm.aggregate(t, reduction=AggregationOp.MEAN)
    print(f"[rank {rank}] after aggregate : {t.tolist()}  (expected all 1.0)")

    comm.close()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)
    print("communicator demo finished OK")
