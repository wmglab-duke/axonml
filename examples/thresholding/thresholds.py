import argparse

import numpy as np
import torch

from axonml.instruments.thresholder import Thresholder
from axonml.models import SMF


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--field", choices=["imthera", "livanova"], help="Cuff.", required=True)

parser.add_argument(
    "-p", "--preload", action="store_true", help="Preload bases array into memory.", default=True
)

parser.add_argument(
    "-v",
    "--visualize",
    action="store_true",
    help="Plot predicted thresholds & error histogram.",
)

args = parser.parse_args()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    # load field and diameter data

    field = args.field

    directory = f"./example_{field}"

    diams = np.load(f"{directory}/example_diameters_{field}.npy")
    n = len(diams)
    fp = np.memmap(
        f"{directory}/example_field_array_{field}.mmap",
        dtype="float32",
        mode="r",
        shape=(n, 1000, 101),
    )
    if args.preload:
        fp = np.array(fp)

    nrn_thresh_path = f"{directory}/example_thresholds_{field}.npy"
    thresh_nrn = np.load(nrn_thresh_path).flatten()

    mrg = SMF(handle_nan=True).cuda().load("MRG")
    thresholder = Thresholder(mrg, fp, diams).float()
    thresh, _ = thresholder.calculate_thresholds()

    err = 100 * (thresh - thresh_nrn) / thresh_nrn

    print(f"mean % threshold error: {err.mean()}%")
    print(f"mean absolute % threshold error: {np.abs(err).mean()}%")
    print(f"min error: {err.min()}%, max error: {err.max()}%")

    # visualize ?
    if args.visualize:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TKAgg")

        lim = np.linspace(0, max(thresh), 100)
        plt.plot(lim, lim, color="grey", alpha=0.6)
        plt.scatter(thresh, thresh_nrn, s=2)
        plt.xlabel("predicted threshold (mA)")
        plt.ylabel("NEURON threshold (mA)")
        plt.axis("square")
        plt.show()

        # error distrubtion
        plt.hist(err)
        plt.xlabel("% threshold error")
        plt.show()
