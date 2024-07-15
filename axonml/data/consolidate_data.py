import argparse

import h5py
import natsort
from tqdm import tqdm

import config


parser = argparse.ArgumentParser()
required = parser.add_argument_group("required named arguments")
required.add_argument("-f", "--filename", help="Input file name", required=True)
required.add_argument("-t", "--to", help="Consolidated file name", required=True)
args = parser.parse_args()

fname = args.filename

sample_size = config.batch_size


state_dict = {}


with h5py.File(fname, "r") as dset:
    n_axons = dset[list(dset.keys())[0]]["activations"].shape[1]
    keys = natsort.natsorted(list(dset.keys()))
    state_vars = list(dset[keys[0]].keys())
    with h5py.File(args.to, "w") as new:
        for state in state_vars:
            state_dict[state] = new.create_dataset(
                state,
                (len(keys), *dset[keys[0]][state].shape),
                chunks=(1, *dset[keys[0]][state].shape),
            )
        ys = new.create_dataset(
            "ys", (len(keys), sample_size, n_axons), chunks=(1, sample_size, n_axons)
        )
        d = new.create_dataset(
            "diameters",
            (len(keys), sample_size, n_axons),
            chunks=(1, sample_size, n_axons),
        )
        for i, k in enumerate(tqdm(keys)):
            for state in state_vars:
                state_dict[state][i, ...] = dset[k][state][:]
            d[i, :, :] = dset[k]["parameters"][:, 0 : 2 * n_axons : 2]
            ys[i, :, :] = dset[k]["parameters"][:, 1 : (2 * n_axons) + 1 : 2]
