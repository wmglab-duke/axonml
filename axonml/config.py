"""Training config."""
from typing import List

import torch

from axonml.models import Backend as A, Axon, MRG


# model specification
model: Axon = MRG
cuda = torch.cuda.is_available()
fp32 = False
nodes = 53
dt: float = A.dt

# training / validation data locations
train_dset = "./data/example_datasets/train_example.h5"
valid_dset = "./data/example_datasets/valid_example.h5"

# states
states = ["m", "h", "p", "s", "v"]

# -- training params --
epochs = 5
truncation_length = 50
lr = 3e-5
grad_accumulation = False
to_train: List[str] = [
    "conductances",
    "membrane",
    "axon_d",
    "node_d",
    "aq10_1",
    "pq10_1",
    "aq10_2",
    "aq10_3",
    "ssd",
]

# -- data --
train_n_idx = 64
val_n_idx = 8
train_chunk_size = 1
val_chunk_size = 1

# -- downsampling --
sampling: int = None

# -- want to observe model parameter values while training? --
postfix: List[str] = ["cm"]

# -- saving & checkpointing
save_every: int = 1
save_dir = "checkpoints"
