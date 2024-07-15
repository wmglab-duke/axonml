import h5py
import torch

import config
from axonml.models import Axon
from axonml.training import tbptt, DataLoader


def collect_states(h5py_file, states):
    return [h5py_file[s] for s in states]


dset_t = h5py.File(config.train_dset, "r")
dset_v = h5py.File(config.valid_dset, "r")


if __name__ == "__main__":
    if not config.fp32:
        torch.set_default_dtype(torch.float64)

    model: Axon = config.model(fp32=config.fp32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss = torch.nn.MSELoss()

    if config.cuda:
        model = model.cuda()
        loss = loss.cuda()

    for p in config.to_train:
        model.unfreeze(p)

    train_data = DataLoader(
        config.nodes,
        dset_t["e"],
        collect_states(dset_t, config.states),
        dset_t["diameters"],
        config.train_n_idx,
        config.train_chunk_size,
        config.sampling,
        model.device(),
        config.fp32,
    )

    val_data = DataLoader(
        config.nodes,
        dset_v["e"],
        collect_states(dset_v, config.states),
        dset_v["diameters"],
        config.val_n_idx,
        config.val_chunk_size,
        config.sampling,
        model.device(),
        config.fp32,
    )

    # perform training
    tbptt(
        model,
        train_data,
        loss,
        optimizer,
        config.epochs,
        config.truncation_length,
        config.dt,
        config.grad_accumulation,
        config.postfix,
        config.save_every,
        config.save_dir,
        val_data,
    )
