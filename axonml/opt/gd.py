"""
Methods and classes for gradient-based parameter optimization
using differentiable neural surrogate model.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from pytorch_optimizer import Ranger
import torch
from tqdm import trange

from cajal.nrn.sources import PreComputedInterpolate1D
from cajal.nrn.stimuli import Stimulus, MonophasicPulse
from cajal.opt.loss import PredictionLoss

from axonml.models.callbacks import Recorder


class WeightedQuotient(torch.nn.Module):
    def __init__(self, target, weights, scale=1):
        super(WeightedQuotient, self).__init__()
        self.target = torch.nn.Parameter(target, requires_grad=False)
        self.off_target = torch.nn.Parameter(1 - self.target, requires_grad=True)
        self.weights = torch.nn.Parameter(weights, requires_grad=False)
        self.scale = scale

    def forward(self, x):
        x = x[:, :, 0, :]
        x = torch.sum(x[:, :, :10], (0, 2)) + torch.sum(x[:, :, 90:], (0, 2))
        x = x * self.weights
        return self.scale * (x @ self.off_target / x @ self.target)


class WeightedBinaryCrossEntropy(PredictionLoss):
    def __init__(self, target, weights):
        super().__init__(target)
        self.weights = weights

    def loss(self, target, predicted):
        target = np.asarray(target)
        predicted = np.asarray(predicted).flatten()
        term_0 = (1 - target) * np.log(1 - predicted + 1e-7)
        term_1 = target * np.log(predicted + 1e-7)
        return -np.average((term_0 + term_1), axis=0, weights=self.weights)


@dataclass
class FieldSpec:
    fields: List[np.array]
    fiber_z: np.array
    nc: int

    def midpoint(self):
        return self.fiber_z.min() + (np.ptp(self.fiber_z) / 2)


@dataclass
class AxonSpec:
    diameter: float
    nodes: int
    length: float

    def dx(self):
        half_length = self.length / 2
        return np.linspace(-half_length, half_length, self.nodes)


class GDProblem:
    def __init__(self, target, weights, f_spec: FieldSpec, a_spec: AxonSpec, time):
        self.f_spec = f_spec
        self.a_spec = a_spec
        self.target = target
        self.weights = weights
        self.time = time

        self.fields = f_spec.fields
        self.fiber_z = f_spec.fiber_z
        self.nc = f_spec.nc

        self.diameter = a_spec.diameter
        self.nodes = a_spec.nodes
        self.length = a_spec.length

        self.n_axons = self.fields[0].shape[0]

        interpolators = [
            PreComputedInterpolate1D(
                field,
                self.fiber_z,
                in_memory=True,
                method="linear",
                fill_value="point_source",
                truncate=0.2,
            )
            for field in self.fields
        ]

        interp_at = a_spec.dx() + f_spec.midpoint()

        all_bases = []

        for i in range(self.n_axons):
            bases_ = []
            for interp in interpolators:
                b = interp._interpolate(interp.get(i), interp.x[0], interp_at)
                bases_.append(b)
            all_bases.append(np.array(bases_))

        all_bases = np.stack(all_bases)

        self.bases = torch.Tensor(all_bases).double().cuda()
        self.diams = self.diameter * torch.ones(self.n_axons).cuda().double()

    def x_to_input(self) -> torch.Tensor:
        raise NotImplementedError()


class GDProblemUniform(GDProblem):
    def __init__(self, target, weights, f_spec, a_spec, time, stim: Stimulus):
        super().__init__(target, weights, f_spec, a_spec, time)
        self.stim = torch.Tensor(stim.timecourse(self.time)).double().cuda()
        self.x = torch.zeros(1, self.nc, requires_grad=True, device="cuda").double()
        self.x.retain_grad()
        self.ndim = torch.numel(self.x)

    def x_to_input(self):
        input = torch.einsum("t,pb,abn -> tpan", self.stim, self.x, self.bases).reshape(
            -1, self.n_axons, self.nodes
        )
        input = input.unsqueeze(2)
        return input


class GDProblemArbitrary(GDProblem):
    def __init__(self, target, weights, f_spec, a_spec, time, pw, delay, dt=0.005):
        super().__init__(target, weights, f_spec, a_spec, time)

        # arbitrary stimulus is nonzero at ...
        self.pw = pw
        self.delay = delay
        self.dt = dt

        # generate mask
        sb = MonophasicPulse(1, pw, delay)
        tcourse = sb.timecourse(self.time)
        self.mask = torch.Tensor(tcourse).double().cuda()

        # to be optimized
        self.x = torch.zeros(
            len(self.time), 1, self.nc, requires_grad=True, device="cuda"
        ).double()
        self.x.retain_grad()

        # size of problem
        self.ndim = self.nc * int(pw / dt)

    def x_to_input(self):
        x_input = self.x - torch.mean(
            self.x[int(self.delay / self.dt) : int((self.delay + self.pw) / self.dt)], 0
        )
        x_input = x_input * self.mask[:, None, None]
        input = torch.einsum("tpb,abn -> tpan", x_input, self.bases).reshape(
            -1, self.n_axons, self.nodes
        )
        input = input.unsqueeze(2)
        return input


class GD:
    rec = Recorder()

    xs = []
    preds = []

    def __init__(self, problems: List[GDProblem], lr, lr_decay):
        self.problems = problems
        self.lr = lr
        self.lr_decay = lr_decay

        self.best_loss = [None] * len(problems)
        self.best_torch_loss = [None] * len(problems)
        self.best_xs = [[] for _ in range(len(problems))]
        self.best_preds = [[] for _ in range(len(problems))]

        self.loss_fns = [
            WeightedQuotient(
                torch.Tensor(p.target), torch.Tensor(p.weights), np.sqrt(p.ndim / p.nc)
            )
            .cuda()
            .double()
            for p in self.problems
        ]

        self.wds = [0.01 * p.n_axons for p in self.problems]

        self.optimizers = [
            Ranger([p.x], lr=self.lr, weight_decay=wd)
            for p, wd in zip(self.problems, self.wds)
        ]

        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)
            for optimizer in self.optimizers
        ]

        self.loss_nps = [
            WeightedBinaryCrossEntropy(p.target, p.weights) for p in self.problems
        ]

    def best(self):
        return [x[-1] for x in self.best_xs]

    def solve(self, model, steps, dt=0.005):
        problems = self.problems

        diams = torch.concat([p.diams for p in problems])
        n_axon_list = [p.n_axons for p in problems]

        x_list = [p.x for p in problems]

        rec = self.rec

        for step in trange(steps):
            rec.reset()

            inputs = []
            for p in problems:
                inputs.append(p.x_to_input())

            input = torch.concat(inputs, dim=1)
            model.run(input, diams, dt=dt, callbacks=[rec], reinit=True)

            out = rec.stack()

            all_losses = []
            all_out = torch.split(out, n_axon_list, dim=1)

            pred_act = []

            for j, (out, p, loss_np, loss_fn, n_axons) in enumerate(
                zip(all_out, problems, self.loss_nps, self.loss_fns, n_axon_list)
            ):
                with torch.no_grad():
                    active = torch.any(
                        torch.any((out[:, :, -1, [10, 90]] > 0), dim=2), dim=0
                    )
                    active = active.cpu().numpy().reshape(1, n_axons)
                pred_act.append(active)
                np_loss = loss_np.loss(p.target, active)

                loss = loss_fn(out)
                all_losses.append(loss)

                if self.best_loss[j] is None:
                    self.best_loss[j] = np_loss
                    self.best_torch_loss[j] = loss.item()
                    best_x = x_list[j].detach().cpu().numpy()
                    self.best_xs[j].append(best_x)
                    self.best_preds[j].append(active)
                if np_loss < self.best_loss[j]:
                    self.best_loss[j] = np_loss
                    best_x = x_list[j].detach().cpu().numpy()
                    self.best_xs[j].append(best_x)
                    self.best_preds[j].append(active)
                if np_loss == self.best_loss[j]:
                    if loss.item() < self.best_torch_loss[j]:
                        self.best_torch_loss[j] = loss.item()
                        self.best_loss[j] = np_loss
                        best_x = x_list[j].detach().cpu().numpy()
                        self.best_xs[j].append(best_x)
                        self.best_preds[j].append(active)
                        if self.best_loss[j] < 1.0:
                            self.schedulers[j].step()

            loss = sum(all_losses)
            loss.backward()
            for x, n in zip(x_list, n_axon_list):
                torch.nn.utils.clip_grad_norm_(x, 200 / n)

            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
