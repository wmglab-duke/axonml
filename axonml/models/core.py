import math
from typing import List, Tuple, Optional

import torch
import torch.jit as jit
from torch import Tensor

from axonml import trained
from .callbacks import CallbackList, Callback
from .backend import Backend as A


def gen_dict_extract(key, var):
    """Utility function to retrieve values from nested dictionaries."""
    for k, v in var.items():
        if k == key:
            yield v
        if isinstance(v, dict):
            for result in gen_dict_extract(key, v):
                yield result


def _inherit_constants(superclass, constants):
    return superclass.__constants__ + constants


class SymmetricConv1D(torch.nn.Conv1d):
    def forward(self, x):
        weight_ = (self.weight + torch.flip(self.weight, [-1])) / 2
        return self._conv_forward(x, weight_, self.bias)


class Axon(jit.ScriptModule):
    """Base 1D fiber class."""

    params = {}

    __constants__ = ["fp32", "temp", "handle_nan", "n_states"]

    def __init__(self, temp=37.0, fp32=True, handle_nan=False, n_states=0):
        super().__init__()

        self.temp = temp
        self.fp32 = fp32
        self.handle_nan = handle_nan
        self.is_cuda = False
        self.n_states = n_states

        # solver stuff
        weight = [1.0, -2.0, 1.0]
        self.ssd = SymmetricConv1D(
            2, 1, 3, bias=False, padding="same", padding_mode="reflect"
        )
        self.ssd.weight.data = torch.tensor([weight, weight]).reshape(1, 2, 3)
        for p in self.ssd.parameters():
            p.requires_grad = False

        self.state = torch.tensor(0)

        # -- constants --
        self.pi = torch.nn.Parameter(torch.tensor(math.pi, requires_grad=False))

        # -- parameters --
        for group, gdict in self.__class__.params.items():
            setattr(self, group, [])
            for pname, pval in gdict.items():
                setattr(
                    self,
                    pname,
                    torch.nn.Parameter(torch.tensor(pval), requires_grad=False),
                )
                getattr(self, group).append(getattr(self, pname))

        self.eval()

    def cuda(self, device=None):
        self.is_cuda = True
        return super().cuda(device)

    def device(self):
        return "cuda" if self.is_cuda else "cpu"

    def unfreeze(self, name=None):
        """Specify that a parameter, parameter group, or module
        will be trained.

        Parameters
        ----------
        name: str, optional
            Name of parameter, group, or module that will be trained,
            by default None. If None, all parameters will be unfrozen.
        """
        if name is not None:
            entity = getattr(self, name)
            if isinstance(entity, list):
                for p in entity:
                    p.requires_grad = True
            elif isinstance(entity, torch.nn.Module):
                for p in entity.parameters():
                    p.requires_grad = True
            else:
                entity.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def freeze(self, name=None):
        """Specify that a parameter, parameter group, or module
        not be trained.

        Parameters
        ----------
        name: str, option
            Name of parameter, group, or module that will not be trained,
            by default None. If None, all parameters will be frozen.
        """
        if name is not None:
            entity = getattr(self, name)
            if isinstance(entity, list):
                for p in entity:
                    p.requires_grad = False
            elif isinstance(entity, torch.nn.Module):
                for p in entity.parameters():
                    p.requires_grad = False
            else:
                entity.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = False

    def reset(self, name=None):
        """Reset parameter, parameter group, or module to defaults.

        Parameters
        ----------
        name : str, optional
            Name of parameter, group, or module to be reset, by default None.
            If None, all parameters will be reset.
        """
        if name is None:
            for _, gdict in self.__class__.params.items():
                for pname, pval in gdict.items():
                    p = getattr(self, pname)
                    p.data = torch.tensor(pval, device=p.device)
            self.ssd.weight.data = torch.tensor(
                [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]], device=self.ssd.weight.device
            ).reshape(1, 2, 3)

        else:
            if name == "ssd":
                self.ssd.weight.data = torch.tensor(
                    [[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]], device=self.ssd.weight.device
                ).reshape(1, 2, 3)

            elif name in self.__class__.params:
                for pname, pval in self.__class__.params[name]:
                    p = getattr(self, pname)
                    p.data = torch.tensor(pval, device=p.device)

            else:
                p = getattr(self, name)
                p.data = torch.tensor(
                    list(gen_dict_extract(name, self.__class__.params))[0],
                    device=p.device,
                )

    def set(self, key, value):
        """Set the value of a specific parameter.

        Parameters
        ----------
        key : str
            Name of parameter to set.
        value : float
            Value of parameter to be set.
        """
        p = getattr(self, key)
        if isinstance(p, Tensor):
            p.data = torch.tensor(value, dtype=p.data.dtype, device=p.device)
        else:
            setattr(self, key, value)

    def cnexp(self, gv, inf, tau_inv, dt: float) -> Tensor:
        """Calculate 1 timestep update to gating variable.

        Parameters
        ----------
        gv : Tensor
            Gating variable at t - 0.5dt
        inf : torch.Tensor
            Gating variable infinity value for vm at time t.
        tau_inv : torch.Tensor
            1 / Gating variable time constant for vm at time t.
        dt : float
            timestep

        Returns
        -------
        torch.Tensor
            Gating variable at time t + 0.5dt
        """
        return inf - (inf - gv) * torch.exp(-dt * tau_inv)

    def dv(self, cm, ra, d2v, ion, dt: float) -> Tensor:
        """Calculate dv/dt

        Parameters
        ----------
        cm : torch.Tensor
            Node capacitance.
        ra : torch.Tensor
            Internodal resistance.
        d2v : torch.Tensor
            Spatial 2nd difference.
        ion : torch.Tensor
            Ionic current.
        dt : float
            Timestep.

        Returns
        -------
        Tensor
            dv/dt
        """
        cm = cm[:, None, None]
        ra = ra[:, None, None]
        dv = dt * (1 / cm) * (((1 / ra) * d2v) - ion)
        if self.fp32:
            return dv.float()
        return dv

    def currents(
        self,
        states: List[Tensor],
        gbar: List[Tensor],
        i: int,
        intra: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate membrane currents.

        Parameters
        ----------
        states : List[Tensor]
            List of system states (gating variables and Vm) for
            all compartments.
        gbar : List[Tensor]
            List of maximum conductance values for all ionic currents.
        i : int
            Index of timestep.
        intra : Optional[Tensor], optional
            Intracellular current, by default None

        Returns
        -------
        Tensor
            Membrane currents for all compartments.
        """

        c = self.ionic_currents(states, gbar)
        if intra is not None:
            c -= intra[i]
        return c

    def set_state(self, state: Optional[Tensor]):
        """Set the system state.

        Parameters
        ----------
        state : Optional[Tensor]
            System state (gating variables and Vm)
        """
        if state is not None:
            self.state = state

    def prepare_inputs(
        self, n_states: int, ic: Optional[Tensor], ve: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        self.set_state(ic)
        states = torch.chunk(self.state, n_states, 1)
        ves = ve.unbind(0)
        return states, ves

    def ionic_currents(self, states, gbar):
        raise NotImplementedError

    def prepare_parameters(
        self, diameters: Tensor
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor]]:
        raise NotImplementedError

    def update_gvs(self, states: List[Tensor], dt: float) -> List[Tensor]:
        raise NotImplementedError

    @jit.script_method
    def advance(
        self,
        ves: List[Tensor],
        states: List[Tensor],
        gbar: List[Tensor],
        cm: Tensor,
        ra: Tensor,
        intra: Optional[Tensor],
        dt: float,
        i: int,
        vm_index: int = -1,
    ) -> List[Tensor]:
        # -- 2nd diff --
        ve = ves[i]
        x = torch.cat([states[vm_index], ve], dim=1)
        d2vdx2 = self.ssd(x)

        # -- update gvs --
        states = self.update_gvs(states, dt)

        # -- calculate ionic current --
        i_ion = self.currents(states, gbar, i, intra)

        # -- update vm --
        dv = self.dv(cm, ra, d2vdx2, i_ion, dt)
        vm = self.advance_vm(states[vm_index], dv)
        states[vm_index] = vm

        return states

    def ic(self):
        raise NotImplementedError

    @jit.script_method
    def advance_vm(self, vm: Tensor, dv: Tensor) -> Tensor:
        if self.handle_nan:
            vm = torch.nan_to_num(vm + dv, posinf=1e5, neginf=-1e5)
        else:
            vm = vm + dv
        return vm

    def forward(
        self,
        ve: Tensor,
        diameters: Tensor,
        ic: Optional[Tensor] = None,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
    ):
        return self.run(ve, diameters, ic, dt, intra, callbacks, reinit)

    def run(
        self,
        ve: Tensor,
        diameters: Tensor,
        ic: Optional[Tensor] = None,
        dt: float = None,
        intra: Optional[Tensor] = None,
        callbacks: List[Callback] = None,
        reinit: bool = False,
    ):
        """Main run method.

        Parameters
        ----------
        ve : Tensor
            Extracellular potential array (T, B, 1, W)
        diameters : Tensor
            Fiber diameters (um)
        ic : Tensor, optional
            Initial condition (n_axons, n_states, n_nodes) or
            (1, n_states, 1), by default None
        dt : float, optional
            Simulation timestep, by default 0.005
        intra : Tensor, optional
            Intracellular current (nA), by default None
        callbacks : List[Callback], optional
            List of callbacks to execute, by default None
        reinit : bool, optional
            Start from original initial condition, by default False

        Returns
        -------
        int
            1
        """

        with torch.set_grad_enabled(self.training):
            dt = dt if dt is not None else A.dt
            
            if callbacks:
                for c in callbacks:
                    c.dt = dt

            ve = torch.as_tensor(ve, device=self.device())
            diameters = torch.as_tensor(diameters, device=self.device())

            if (ic is None and self.state.ndim == 0) or reinit:
                ic = self.ic()
            if ic is not None:
                ic = torch.as_tensor(ic, device=self.device())
                ic = ic.expand(ve.shape[1], -1, ve.shape[-1])

            states, ves = self.prepare_inputs(self.n_states, ic, ve)
            cm, ra, gbar = self.prepare_parameters(diameters)

            callbacks = CallbackList(callbacks)
            callbacks.pre_loop_hook(self.state)

            for i in range(len(ves)):
                states = self.advance(ves, states, gbar, cm, ra, intra, dt, i)
                self.state = torch.cat(states, dim=1)

                callbacks.post_step_hook(self.state)

            callbacks.post_loop_hook(self.state)
            self.state = self.state.detach()

        return 1

    def compile(self, nodes=16, axons=1):
        ve = torch.ones(1, axons, 1, nodes, device=self.device())
        d = 10 * torch.ones(axons, device=self.device())
        for _ in range(5):
            self.run(ve, d, reinit=True)
        self.state = torch.tensor(0)
        return self

    def load(self, state_dict):
        if state_dict in trained:
            state_dict = torch.load(trained[state_dict], map_location=self.device())
        elif isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location=self.device())
        self.load_state_dict(state_dict)
        return self
