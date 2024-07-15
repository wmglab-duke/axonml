"""
Dataset generator for Multifascicular models instrumented with
ImThera (6 Contact) cuff.

Range of diameters.

Author: Minhaj Hussain
"""

import argparse

import numpy as np

from cajal.mpi import NeuronModel, RNG
from cajal.common.logging import logger, DisableLogger
from cajal.nrn import Backend as N
from cajal.nrn.cells import MRG
from cajal.nrn.sources import RANDPreComputedInterpolate1D
from cajal.nrn.specs import Mutable as Mut
from cajal.nrn.stimuli import MonophasicPulse

import config


parser = argparse.ArgumentParser()

parser.add_argument("--filename", type=str, default=config.filename)

parser.add_argument("--n_batches", type=int, default=config.n_batches)
parser.add_argument("--batch_size", type=int, default=config.batch_size)
parser.add_argument(
    "--axons_per_minibatch", type=int, default=config.axons_per_minibatch
)

parser.add_argument("--round", type=int, default=config.round)
parser.add_argument("--mode", type=str, default="w")
parser.add_argument("--record_v", default=config.record_v, action="store_true")
parser.add_argument("--record_e", default=config.record_e, action="store_true")

parser.add_argument("--diameter", type=float, default=config.diameter)
parser.add_argument("--use_dr", default=config.use_dr, action="store_true")
parser.add_argument("--min_diam", type=float, default=config.min_diam)
parser.add_argument("--max_diam", type=float, default=config.max_diam)

parser.add_argument("--max_amp", type=float, default=config.max_amp)
parser.add_argument("--max_pw", type=float, default=config.max_pw)
parser.add_argument("--max_delay", type=float, default=config.max_delay)

args = parser.parse_args()


N.tstop = config.tstop

N_NODE_RECORD = config.n_node_record
N_CONTACTS = config.n_contacts

DATA_0 = [np.load(f"./fields/P/{i}.npy") for i in range(N_CONTACTS)]
DATA_1 = [np.load(f"./fields/H/{i}.npy") for i in range(N_CONTACTS)]
DATA_ALL = [np.vstack([x, y]) for x, y in zip(DATA_0, DATA_1)]

fiber_zs = np.load("./fields/H/fiber_z.npy")

midpoint = fiber_zs.min() + (np.ptp(fiber_zs) / 2)

N_AXONS = args.axons_per_minibatch


class RandMonophasic(MonophasicPulse):
    def __init__(self, amp, pw, delay):
        amp = amp[0] + (amp[1] - amp[0]) * np.random.rand()
        pw = pw[0] + (pw[1] - pw[0]) * np.random.rand()
        delay = delay[0] + (delay[1] - delay[0]) * np.random.rand()
        super(RandMonophasic, self).__init__(amp, pw, delay)


class MRGDynamicsRecorder(NeuronModel):
    def recording(self, axons):
        params = ["mp", "m", "h", "s"]
        for p in params:
            self.record(
                f"{p}_axnode_myel",
                p[-1],
                [a.middle(N_NODE_RECORD, "node") for a in axons],
                N_NODE_RECORD,
            )
        if args.record_v:
            self.record(
                "v",
                "v",
                [a.middle(N_NODE_RECORD, "node") for a in axons],
                N_NODE_RECORD,
            )
        if args.record_e:
            self.record(
                "e_extracellular",
                "e",
                [a.middle(N_NODE_RECORD, "node") for a in axons],
                N_NODE_RECORD,
            )


axons_list = [
    MRG.SPEC(
        diameter=Mut(),
        y=Mut(),
        gid=i,
        enforce_odd_axonnodes=True,
        axonnodes=N_NODE_RECORD,
        try_exact=False,
        passive_end_nodes=False,
        piecewise_geom_interp=False,
        interpolation_method=1,
    )
    for i in range(N_AXONS)
]

max_amp = args.max_amp
max_pw = args.max_pw
max_delay = args.max_delay

extra_list = [
    RANDPreComputedInterpolate1D.SPEC(
        data,
        y=fiber_zs,
        in_memory=True,
        method="linear",
        fill_value="point_source",
        truncate=0.05,
    )
    << RandMonophasic.SPEC(
        amp=(-max_amp, max_amp), pw=(0, max_pw), delay=(0, max_delay)
    )
    for data in DATA_ALL
]

with_v = "with_v" if args.record_v else "without_v"
with_e = "with_e" if args.record_e else "without_e"

min_diam = args.min_diam
max_diam = args.max_diam

if args.use_dr:
    ftype = f"MRG{min_diam}to{max_diam}"
else:
    ftype = f"MRG{args.diameter}"

if args.filename is not None:
    fname = args.filename
else:
    fname = f"""{ftype}-random_fields_monophasic_{max_amp}mA_
                {with_v}_{with_e}_tstop{N.tstop_}ms_pw{max_pw}ms_
                delay{max_delay}ms-{args.round}.h5"""

model = MRGDynamicsRecorder(axons_list, extra_list).mpio(fname, args.mode)


def run():
    batch_size = args.batch_size
    for i in range(args.n_batches):
        if args.use_dr:
            diameters = min_diam + RNG.random(N_AXONS * batch_size) * (
                max_diam - min_diam
            )
            deltas = np.array(
                [
                    MRG.geometric_params(d, piecewise=False, try_exact=False, interp=1)[
                        "deltax"
                    ]
                    for d in diameters
                ]
            ).reshape(-1, N_AXONS)
            diameters = diameters.reshape(-1, N_AXONS)
        else:
            diameters = args.diameter * np.ones(N_AXONS * batch_size)
            deltas = (
                MRG.geometric_params(args.diameter, piecewise=False, try_exact=False)[
                    "deltax"
                ]
                * np.ones(N_AXONS * batch_size)
            ).reshape(-1, N_AXONS)
            diameters = diameters.reshape(-1, N_AXONS)

        ys = (
            midpoint
            - (deltas / 2)
            + (RNG.random(N_AXONS * batch_size).reshape(-1, N_AXONS) * deltas)
        )
        geometric_params = np.dstack((diameters, ys)).reshape(diameters.shape[0], -1)
        model.run(geometric_params)


with DisableLogger(logger):
    run()

model.mpio_close()
