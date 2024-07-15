import argparse
import datetime

import numpy as np
import torch

from cajal.mpi import ANNGPURunner
from cajal.nrn.stimuli import SymmetricBiphasic
from cajal.nrn.sources import PreComputedInterpolate1D
from cajal.opt.differentialevolution import DEBASE
from cajal.opt.differentialevolution.callbacks import Logger, EarlyStopping, Timer

from axonml.models import SMF
from axonml.models.callbacks import Active

import de_config as config
from utils import (
    percent_off_target_active,
    percent_on_target_active,
    deltax,
    WeightedBinaryCrossEntropy,
)


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--sample")

# DE params
parser.add_argument("--strategy", default=config.strategy)
parser.add_argument("--mutation", type=float, default=config.mutation)
parser.add_argument("--recombination", type=float, default=config.recombination)
parser.add_argument(
    "--parameter-adaptation", type=float, default=config.parameter_adaptation
)
parser.add_argument("--popsize", type=int, default=config.popsize)
parser.add_argument("--maxiter", type=int, default=config.maxiter)
parser.add_argument("--lb", type=float, default=config.lb)
parser.add_argument("--ub", type=float, default=config.ub)
parser.add_argument("--init", default=config.init)
parser.add_argument("--seed", nargs="?", type=int, const=config.seed)
parser.add_argument("--diameter", type=float, default=config.diameter)
parser.add_argument("--nodes", type=int, default=config.nodes)
parser.add_argument("--nc", type=int, default=config.nc)

# validate with NEURON?
parser.add_argument("--validate", action="store_true")
args = parser.parse_args()


# -- utility functions & class definitions --


def x_to_input_uniform_stim(x, bases, stim, popsize, n_axons, nodes):
    """
    x: (popsize, n_bases) torch.Tensor
    bases: (n_axons, n_bases, n_nodes) torch.Tensor
    stim: (n_timesteps) torch.Tensor

    -> T, B, C, W
    """
    input = torch.einsum("t,pb,abn -> tpan", stim, x, bases).reshape(
        -1, popsize * n_axons, nodes
    )
    input = input.unsqueeze(2)
    return input


# -- fields --

nodes = args.nodes
diameter = args.diameter

sample = args.sample

nc = args.nc

fields = [np.load(f"./samples/{sample}/{i}.npy") for i in range(nc)]
fiber_zs = np.load(f"./samples/{sample}/fiber_z.npy")
target = np.load(f"./samples/{sample}/target.npy").astype(bool)
weights = np.load(f"./samples/{sample}/weights.npy")

midpoint = fiber_zs.min() + (np.ptp(fiber_zs) / 2)

n_axons = fields[0].shape[0]

interpolators = [
    PreComputedInterpolate1D(
        field,
        fiber_zs,
        in_memory=True,
        method="linear",
        fill_value="point_source",
        truncate=0.2,
    )
    for field in fields
]

start = (deltax(diameter) * (nodes - 1)) / 2
dx = np.linspace(-start, start, nodes)
interp_at = np.linspace(-start, start, nodes) + midpoint

all_bases = []

for i in range(n_axons):
    bases_ = []
    for interp in interpolators:
        b = interp._interpolate(interp.get(i), interp.x[0], interp_at)
        bases_.append(b)
    all_bases.append(np.array(bases_))


all_bases = np.stack(all_bases)
bases = torch.Tensor(all_bases).float().cuda()


# -- model --

mrg = SMF().cuda().load("MRG").compile(nodes=nodes)


# -- stim --

sb = SymmetricBiphasic(1, 0.4, 0.5)
tcourse = sb.timecourse(np.arange(0, 5, 0.005))
stim = torch.Tensor(tcourse).float().cuda()


# -- loss --

loss = WeightedBinaryCrossEntropy(target, weights)


# -- instrumentation to run predictions --

popsize = args.popsize
hbs = popsize
diams = diameter * torch.ones(n_axons * hbs).cuda().float()


class Predictor(ANNGPURunner):
    rec = Active(t_start_check=0.5, node_check=[5, -5])

    def func(self, x, **kwargs):
        self.rec.reset()
        x = torch.Tensor(x).float().cuda()
        with torch.no_grad():
            input = x_to_input_uniform_stim(x, bases, stim, hbs, n_axons, nodes)
            mrg.run(input, diams, callbacks=[self.rec], dt=0.005, reinit=True)
        active = self.rec.numpy().reshape(hbs, n_axons)
        return np.array([loss.loss(target, i) for i in active])


pred = Predictor([(1,)], "f")


if __name__ == "__main__":
    time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S")
    savedir = f"./de_results/sample_{sample}_{time}/"
    timer = Timer()
    callbacks = [Logger(savedir), EarlyStopping(min_loss=1e-7), timer]
    de = DEBASE(
        pred,
        bounds=[(args.lb, args.ub)] * 6,
        updating="deferred",
        maxiter=args.maxiter,
        popsize=popsize,
        strategy=args.strategy,
        parameter_adaptation=args.parameter_adaptation,
        mutation=args.mutation,
        recombination=args.recombination,
        boundary_constraint="resample",
        callbacks=callbacks,
        init=args.init,
        seed=args.seed,
    )
    hist = de.solve()
    with open(savedir + "time.txt", "w") as f:
        f.write(timer.total_time_str)
    hist.save(savedir + "history.hist")

    # validate with NEURON
    if args.validate:
        from cajal.nrn import MRG
        from cajal.mpi import NeuronModel
        from cajal.nrn.specs import Mutable as Mut

        class MyMRG(MRG):
            def init_AP_monitors(self):
                self.set_AP_monitors(axonnodes=[5, -5], threshold=0)

        axons = []
        for i in range(fields[0].shape[0]):
            axons.append(
                MyMRG.SPEC(
                    y=midpoint,
                    gid=i,
                    enforce_odd_axonnodes=True,
                    axonnodes=nodes,
                    diameter=diameter,
                    try_exact=False,
                    piecewise_geom_interp=False,
                    interpolation_method=1,
                    passive_end_nodes=False,
                )
            )

        # stim
        extra = [
            PreComputedInterpolate1D.SPEC(
                field,
                fiber_zs,
                in_memory=True,
                method="linear",
                fill_value="point_source",
                truncate=0.2,
            )
            << SymmetricBiphasic.SPEC(amp=Mut(), pw=0.4, delay=0.5)
            for field in fields
        ]

        # build model
        model = NeuronModel(axons, extra_spec=extra, load_balancing="dynamic")
        model.run(hist.history.x[-1])
        active = model.activations.flatten()
        target = target.flatten()
        weights = weights.flatten()
        ont = percent_on_target_active(active, target, weights)
        offt = percent_off_target_active(active, target, weights)
        print(f"{ont:.2f}% on target, {offt:.2f}% off target")
