import pickle
import os

import numpy as np
from natsort import natsorted

from cajal.common.logging import tic, toc
from cajal.nrn.stimuli import SymmetricBiphasic
from cajal.nrn.sources import PreComputedInterpolate1D

from axonml.models import SMF
from axonml.opt.gd import AxonSpec, FieldSpec, GDProblemUniform, GD

from gd_parser import parser
from utils import deltax, percent_on_target_active, percent_off_target_active

args = parser.parse_args()


# -- stim --

stim = SymmetricBiphasic(1, 0.4, 0.4)
tcourse = np.arange(0, 2.5, 0.005)


# -- bases --

nodes = args.nodes
diameter = args.diameter
nc = args.nc

if args.samples == "all":
    all_samples = natsorted([f.name for f in os.scandir("./samples/") if f.is_dir()])
else:
    all_samples = args.samples


# -- load fields, construct problems --

problems = []

for sample in all_samples:
    fields = [np.load(f"./samples/{sample}/{i}.npy") for i in range(nc)]
    fiber_zs = np.load(f"./samples/{sample}/fiber_z.npy")
    target = np.load(f"./samples/{sample}/target.npy").astype(bool)
    weights = np.load(f"./samples/{sample}/weights.npy")

    length = deltax(diameter) * (nodes - 1)

    f_spec = FieldSpec(fields, fiber_zs, nc)
    a_spec = AxonSpec(diameter, nodes, length)

    problem = GDProblemUniform(target, weights, f_spec, a_spec, tcourse, stim)
    problems.append(problem)


# -- model --

mrg = SMF(fp32=False).cuda().load(args.model).compile(nodes=nodes).train().double()


# -- run optimization --

if __name__ == "__main__":
    gd = GD(problems, args.lr, args.lr_decay)

    tic()
    gd.solve(mrg, args.n_steps)
    time = toc()

    if args.validate:
        from cajal.nrn import MRG
        from cajal.mpi import NeuronModel
        from cajal.nrn.specs import Mutable as Mut
        from cajal.nrn import Backend as N

        class MyMRG(MRG):
            def init_AP_monitors(self):
                self.set_AP_monitors(axonnodes=[10, 90])

        N.tstop = 2.5

        predictions = []
        on_target_active = []
        off_target_active = []
        wbces = []

        for ii, (p, best_x) in enumerate(zip(problems, gd.best_xs)):
            fields = p.fields
            fiber_zs = p.fiber_z
            target = p.target
            weights = p.weights

            axons = []
            for i in range(fields[0].shape[0]):
                axons.append(
                    MyMRG.SPEC(
                        y=p.f_spec.midpoint(),
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
                << SymmetricBiphasic.SPEC(amp=Mut(), pw=0.4, delay=0.4)
                for field in fields
            ]

            # build model
            model = NeuronModel(axons, extra_spec=extra, load_balancing="dynamic")
            model.run(best_x[-1])
            pred = model.activations[0]
            predictions.append(pred)
            on_target_active.append(percent_on_target_active(pred, target, weights))
            off_target_active.append(percent_off_target_active(pred, target, weights))
            wbce = gd.loss_nps[ii].loss(target, pred)
            wbces.append(wbce)

        with open("pred_gd_biphasic.pkl", "wb") as f:
            pickle.dump(predictions, f)

        for s, ont, offt, wbce in zip(
            all_samples, on_target_active, off_target_active, wbces
        ):
            print(s, ont, offt, wbce)
