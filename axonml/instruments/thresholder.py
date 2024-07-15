from typing import List, Tuple

import numpy as np

import torch
from torch import Tensor

from axonml.models import Axon
from axonml.models.callbacks import Active, ActiveAL, Recorder


class Thresholder:
    def __init__(
        self,
        model: Axon,
        bases: Tensor,
        diams: Tensor,
        dt=0.005,
        ub=None,
        fix_bound_up=5.0,
        fix_bound_down=0.1,
        max_tries_bound_fix=10,
        max_tries_thresh=25,
        resolution=0.01,
        threshold=0.0,
        node_check: List[int] = [5, -5],
        t_start_check=0.0,
        at_least=1
    ):
        if isinstance(diams, Tensor) or isinstance(diams, np.ndarray):
            assert len(diams) == bases.shape[0]

        elif isinstance(diams, float):
            diams = np.atleast_1d(np.full(bases.shape[0], diams))

        else:
            raise TypeError('diams must be a NumPy array / PyTorch tensor with same length as bases or a float.')
        
        if at_least < 1:
            raise ValueError('at_least must be >= 1.')

        if fix_bound_down >= 1 or fix_bound_up <= 0:
            raise ValueError('fix_bound_down should be < 1 and > 0.')
        
        if fix_bound_up <= 1:
            raise ValueError('fix_bound_up should be > 1.')

        self.model = model.compile(bases.shape[-1], bases.shape[0])

        bases = torch.as_tensor(bases)
        diams = torch.as_tensor(diams)

        self.dt = dt

        bases = bases.permute(1, 0, 2).unsqueeze(2)

        self.bases = bases.to(model.device())
        self.diams = diams.to(model.device())

        self.threshold = threshold

        self.ignore = None

        with torch.no_grad():
            if ub is not None:
                self.ub = ub * torch.ones_like(self.diams, device=model.device())
            else:
                self.ub = 0.2 * torch.ones_like(self.diams) / (self.diams / 5) ** 2
            self.ub_initial = self.ub.clone()
            self.lb = torch.zeros_like(self.ub)

        self.fix_bound_up = fix_bound_up
        self.fix_bound_down = fix_bound_down
        self.max_tries_bound_fix = max_tries_bound_fix
        self.max_tries_thresh = max_tries_thresh
        self.resolution = resolution


        if at_least > 1:
            self.active = ActiveAL(threshold, t_start_check, node_check, dt=dt, at_least=at_least)
        else:
            self.active = Active(threshold, t_start_check, node_check, dt=dt)
        self.rec = Recorder(max_only=True)

    def float(self):
        self.fp32 = True
        self.model = self.model.float()
        self.bases = self.bases.float()
        self.diams = self.diams.float()
        self.ub = self.ub.float()
        self.ub_initial = self.ub.clone()
        self.lb = self.lb.float()
        return self
    
    def double(self):
        self.fp32 = False
        self.model = self.model.double()
        self.bases = self.bases.double()
        self.diams = self.diams.double()
        self.ub = self.ub.double()
        self.ub_initial = self.ub.clone()
        self.lb = self.lb.double()
        return self

    def check_active(self, bound: Tensor):
        """Check whether stimulus amplitudes generates APs.

        Parameters
        ----------
        bound : Tensor
            Amplitudes to test.

        Returns
        -------
        Tensor
            boolean
        """
        self.active.reset()
        ve = self.bases * bound[None, :, None, None]
        self.model.run(ve, self.diams, callbacks=[self.active], reinit=True, dt=self.dt)
        return self.active.is_active()

    def check_active_with_rec(self, bound: Tensor):
        self.active.reset()
        self.rec.reset()
        ve = self.bases * bound[None, :, None, None]
        self.model.run(ve, self.diams, callbacks=[self.active, self.rec], reinit=True, dt=self.dt)
        return self.active.is_active(), self.rec.stack()

    def fix_bounds(self, block_possible=True):
        """Make sure upper bound generates AP."""

        with torch.no_grad():
            tries = 0
            mask, rec = self.check_active_with_rec(self.ub)
            print("Fixing bounds.", end="")
            while torch.any(~mask):
                print(".", end="")
                if tries >= self.max_tries_bound_fix:
                    break
                mask, rec = self.check_active_with_rec(self.ub)
                inactive = ~mask
                if block_possible:
                    self.ub[(rec[:, -1] < self.threshold) & inactive] *= self.fix_bound_up
                    self.ub[(rec[:, -1] >= self.threshold) & inactive] *= self.fix_bound_down
                else:
                    self.ub[inactive] *= self.fix_bound_up
                tries += 1
            else:
                print("Done.")
                return
            print(
                f"Unable to fix bounds within {self.max_tries_bound_fix}"
                " iterations, ignoring some."
            )
            self.ignore = ~mask
            self.ub[self.ignore] = 1
            self.lb[self.ignore] = 1

    def calculate_thresholds(self) -> Tuple[Tensor, Tensor]:
        """Calculate thresholds.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Upper and lower bound on thresholds.
        """
        self.fix_bounds()
        self.rec.reset()
        self.active.reset()

        with torch.no_grad():
            ub = self.ub
            lb = self.lb

            window = (ub - lb) / ub
            msk = window >= self.resolution
            tries = 0

            while torch.any(msk) & (tries < self.max_tries_thresh):
                stimamp = (ub + lb) / 2
                mask = self.check_active(stimamp)
                a_thr = msk & mask
                b_thr = msk & ~mask
                ub[a_thr] = stimamp[a_thr]
                lb[b_thr] = stimamp[b_thr]
                window = (ub - lb) / ub
                msk = window >= self.resolution
                tries += 1
            if tries >= self.max_tries_thresh:
                print("hmm")
                return ub, lb

            # final
            # stimamp = (ub + lb) / 2
            # mask = self.check_active(stimamp)
            # ub[mask] = stimamp[mask]

            if self.ignore is not None:
                ub[self.ignore] = torch.nan
                lb[self.ignore] = torch.nan

            return ub.cpu().numpy(), lb.cpu().numpy()
