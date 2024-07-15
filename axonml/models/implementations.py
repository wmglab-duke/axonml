from typing import List

import torch
from torch import Tensor

from .core import Axon


class SMF(Axon):
    """Surrogate Myelinated Fiber"""
    params = {
        "conductances": {
            "gnabar": 3.0,  # S/cm2
            "gkbar": 0.08,
            "gnapbar": 0.01,
            "gl": 0.007,
        },
        "reversal_potentials": {"ena": 50.0, "ek": -90.0, "el": -90.0},
        "membrane": {
            "cm": 10e-3,  # mF / cm2 - larger for initial stability
            "rhoa": 70.0,  # ohm-cm
        },
        "rate_constants": {
            "ampA": 0.01,
            "ampB": 27.0,
            "ampC": 10.2,
            "bmpA": 0.00025,
            "bmpB": 34.0,
            "bmpC": 10.0,
            "amA": 1.86,
            "amB": 21.4,
            "amC": 10.3,
            "bmA": 0.086,
            "bmB": 25.7,
            "bmC": 9.16,
            "ahA": 0.062,
            "ahB": 114.0,
            "ahC": 11.0,
            "bhA": 2.3,
            "bhB": 31.8,
            "bhC": 13.4,
            "asA": 0.3,
            "asB": -27.0,
            "asC": -5.0,
            "bsA": 0.03,
            "bsB": 10.0,
            "bsC": -1.0,
        },
        "axon_d": {"axond1": 0.0187623, "axond2": 4.787487e-01, "axond3": 1.203613e-01},
        "node_d": {
            "noded1": 6.303781e-03,
            "noded2": 2.070544e-01,
            "noded3": 5.339006e-01,
        },
        "delta_x": {
            "deltax1": -8.215284e00,
            "deltax2": 2.724201e02,
            "deltax3": -7.802411e02,
        },
        "other": {"vtraub": -80.0},
        "q10": {
            "aq10_1": 2.2,
            "pq10_1": 2.2,
            "aq10_2": 2.9,
            "aq10_3": 3.0,
            "bq10_1": 20.0,
            "bq10_2": 20.0,
            "bq10_3": 36.0,
            "cq10_1": 10.0,
            "cq10_2": 10.0,
            "cq10_3": 10.0,
        },
    }

    def __init__(self, temp=37.0, fp32=True, handle_nan=False):
        super().__init__(temp, fp32, handle_nan, 5)

    def ra(self, diameters):
        """Internodal resistance.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Internodal resistance (Ohm)
        """
        radii = diameters / 20000  # radius in cm
        rhoa = self.rhoa * self.rhoa_scale(diameters)
        return (rhoa * self.deltax(diameters)) / (self.pi * (radii**2))

    def rhoa_scale(self, diameters):
        return 1 / ((self.axonD(diameters) / diameters) ** 2)

    def area(self, diameters):
        """Membrane surface area of Node of Ranvier.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Surface area of nodes (cm2)
        """
        lengths = torch.ones_like(diameters) / 10000
        return self.pi * self.nodeD(diameters) * lengths  # cm2

    def axonD(self, diameters):
        """Axon diameter.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Axon diameters (um)
        """
        axond = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
        return axond

    def deltax(self, diameters):
        """Internodal distance.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Axon internodal distances (cm)
        """
        deltax = self.deltax1 * diameters**2 + self.deltax2 * diameters + self.deltax3
        return deltax / 10000

    def nodeD(self, diameters):
        """Node of Ranvier diameter.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Node diameters (cm)
        """
        noded = self.noded1 * diameters**2 + self.noded2 * diameters + self.noded3
        return noded / 10000

    def q10_1(self):
        return self.aq10_1 ** ((self.temp - self.bq10_1) / self.cq10_1)

    def p10_1(self):
        return self.pq10_1 ** ((self.temp - self.bq10_1) / self.cq10_1)

    def q10_2(self):
        return self.aq10_2 ** ((self.temp - self.bq10_2) / self.cq10_2)

    def q10_3(self):
        return self.aq10_3 ** ((self.temp - self.bq10_3) / self.cq10_3)

    def dmdt(self, m, vm, dt: float):
        q10_1 = self.q10_1()
        am = self.alpham(vm)
        bm = self.betam(vm)
        ambm = am + bm
        m_inf = am / ambm
        return self.cnexp(m, m_inf, q10_1 * ambm, dt)

    def dhdt(self, h, vm, dt: float):
        q10_2 = self.q10_2()
        ah = self.alphah(vm)
        bh = self.betah(vm)
        ahbh = ah + bh
        h_inf = ah / ahbh
        return self.cnexp(h, h_inf, q10_2 * ahbh, dt)

    def dpdt(self, p, vm, dt: float):
        q10_1 = self.p10_1()
        ap = self.alphap(vm)
        bp = self.betap(vm)
        apbp = ap + bp
        p_inf = ap / apbp
        return self.cnexp(p, p_inf, q10_1 * apbp, dt)

    def dsdt(self, s, vm, dt: float):
        q10_3 = self.q10_3()
        als = self.alphas(vm)
        bs = self.betas(vm)
        asbs = als + bs
        s_inf = als / asbs
        return self.cnexp(s, s_inf, q10_3 * asbs, dt)

    def alphap(self, vm):
        num = self.ampA * (vm + self.ampB)
        den = 1 - torch.exp(-(vm + self.ampB) / self.ampC)
        b = num / den
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b, nan=self.ampA * self.ampC)
        return b

    def betap(self, vm):
        num = self.bmpA * (-(vm + self.bmpB))
        den = 1 - torch.exp((vm + self.bmpB) / self.bmpC)
        b = num / den
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b, nan=self.bmpA * self.bmpC)
        return b

    def alpham(self, vm):
        num = self.amA * (vm + self.amB)
        den = 1 - torch.exp(-(vm + self.amB) / self.amC)
        b = num / den
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b, nan=self.amA * self.amC)
        return b

    def betam(self, vm):
        num = self.bmA * (-(vm + self.bmB))
        den = 1 - torch.exp((vm + self.bmB) / self.bmC)
        b = num / den
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b, nan=self.bmA * self.bmC)
        return b

    def alphah(self, vm):
        num = self.ahA * (-(vm + self.ahB))
        den = 1 - torch.exp((vm + self.ahB) / self.ahC)
        b = num / den
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b, nan=self.ahA * self.ahC)
        return b

    def betah(self, vm):
        b = self.bhA / (1 + torch.exp(-(vm + self.bhB) / self.bhC))
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b)
        return b

    def alphas(self, vm):
        b = self.asA / (torch.exp((vm + self.asB - self.vtraub) / self.asC) + 1)
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b)
        return b

    def betas(self, vm):
        b = self.bsA / (torch.exp((vm + self.bsB - self.vtraub) / self.bsC) + 1)
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.nan_to_num(b)
        return b

    def ionic_currents(self, states: List[Tensor], gbar: List[Tensor]):
        m, h, p, s, vm = states
        gnabar, gnapbar, gkbar, gl = gbar
        i_ion = (
            (gnabar[:, None, None] * m**3 * h * (vm - self.ena))
            + (gnapbar[:, None, None] * p**3 * (vm - self.ena))
            + (gkbar[:, None, None] * s * (vm - self.ek))
            + (gl[:, None, None] * (vm - self.el))
        )  # mA
        return i_ion

    def prepare_parameters(self, diameters):
        area = self.area(diameters)  # cm2
        cm = self.cm * area  # F
        ra = self.ra(diameters)  # ohm

        gnabar = self.gnabar * area
        gnapbar = self.gnapbar * area
        gkbar = self.gkbar * area
        gl = self.gl * area

        return cm, ra, (gnabar, gnapbar, gkbar, gl)

    def update_gvs(self, states: List[Tensor], dt: float):
        m, h, p, s, vm = states
        m = self.dmdt(m, vm, dt)
        h = self.dhdt(h, vm, dt)
        p = self.dpdt(p, vm, dt)
        s = self.dsdt(s, vm, dt)
        return [m, h, p, s, vm]

    def compile(self, nodes=16, axons=1):
        return super().compile(nodes, axons)

    def ic(self) -> Tensor:
        m = 0.0732093
        h = 0.62069505
        p = 0.20260409
        s = 0.04302994
        v = -80.0

        dtype = None
        if self.fp32:
            dtype = torch.float

        ic = torch.tensor([m, h, p, s, v], dtype=dtype, device=self.device()).reshape(
            1, -1, 1
        )
        return ic
    

class Sundt(Axon):
    params = {
        "conductances": {
            "gnabar": 0.04,  # S/cm2
            "gkbar": 0.04,
            "gl": 0.0001,
        },
        "reversal_potentials": {"ena": 50.0, "ek": -90.0, "el": -65.0},
        "membrane": {
            "cm": 1e-3,  # mF / cm2
            "rhoa": 100.0,  # ohm-cm
        },
        "shifts": {
            "mshift": -6,
            "hshift": 6
        },
        "rate_constants": {
            "am1": 0.32,
            "am2": 13.1,
            "am3": 4.0,
            "ah1": 0.128,
            "ah2": 17.0,
            "ah3": 18.0,
            "bm1": 0.28,
            "bm2": 40.1,
            "bm3": 5.0,
            "bh1": 4.0,
            "bh2": 40.0,
            "bh3": 5.0,
            "bh4": 1.0,
            "zetan": -5.0,
            "zetal": 2.0,
            "gmn": 0.4,
            "gml": 1.0,
            "vhalfn": -32.0,
            "vhalfl": -61.0,
            "a0l": 0.001,
            "a0n": 0.03
        },
        "axon_d": {"axond1": 0.0, "axond2": 1.0, "axond3": 0.0},
        "node_d": {
            "noded1": 0.0,
            "noded2": 1.0,
            "noded3": 0.0,
        },
        "delta_x": {
            "deltax1": 0.0,
            "deltax2": 1.0,
            "deltax3": 0.0,
        },
        "other": {"scale": 1.0},
        "q10": {
            "aq10_1": 3.0,
            "bq10_1": 30.0,
            "cq10_1": 10.0,
            "aq10_2": 3.0,
            "bq10_2": 30.0,
            "cq10_2": 10.0,
            "aq10_3": 3.0,
            "bq10_3": 30.0,
            "cq10_3": 10.0,
            "aq10_4": 3.0,
            "bq10_4": 30.0,
            "cq10_4": 10.0,
        },
    }

    def __init__(self, dx=10, temp=37.0, fp32=True, handle_nan=False):
        super().__init__(temp, fp32, handle_nan, 5)
        self.dx = dx

    def ra(self, diameters):
        """Internodal resistance.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Internodal resistance (Ohm)
        """
        radii = self.axonD(diameters) / 20000  # radius in cm
        rhoa = self.rhoa
        return (rhoa * self.deltax(diameters)) / (self.pi * (radii**2))

    def area(self, diameters):
        """Membrane surface area of Node of Ranvier.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Surface area of nodes (cm2)
        """
        lengths = self.deltax(diameters)
        return self.pi * (self.scale * self.axonD(diameters) / 10000) * lengths  # cm2

    def axonD(self, diameters):
        """Axon diameter.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Axon diameters (um)
        """
        axond = self.axond1 * diameters**2 + self.axond2 * diameters + self.axond3
        return axond

    def deltax(self, diameters):
        """Internodal distance.

        Parameters
        ----------
        diameters : torch.Tensor
            fiber diameters

        Returns
        -------
        torch.Tensor
            Axon internodal distances (cm)
        """
        #deltax = self.deltax1 * self.deltax_**2 + self.deltax2 * self.deltax_ + self.deltax3
        return self.dx * torch.ones_like(diameters) / 10000

    def q10_1(self):
        return self.aq10_1 ** ((self.temp - self.bq10_1) / self.cq10_1)

    def q10_2(self):
        return self.aq10_2 ** ((self.temp - self.bq10_2) / self.cq10_2)

    def q10_3(self):
        return self.aq10_3 ** ((self.temp - self.bq10_3) / self.cq10_3)
    
    def q10_4(self):
        return self.aq10_4 ** ((self.temp - self.bq10_4) / self.cq10_4)

    def dmdt(self, m, vm, dt: float):
        q10 = self.q10_1()
        am = self.alpham(vm)
        bm = self.betam(vm)
        ambm = am + bm
        m_inf = am / ambm
        return self.cnexp(m, m_inf, q10 * ambm, dt)

    def dhdt(self, h, vm, dt: float):
        q10 = self.q10_2()
        ah = self.alphah(vm)
        bh = self.betah(vm)
        ahbh = ah + bh
        h_inf = ah / ahbh
        return self.cnexp(h, h_inf, q10 * ahbh, dt)

    def dndt(self, n, vm, dt: float):
        q10 = self.q10_3()
        an = self.alphan(vm)
        bn = self.betan(vm)
        an_ = (1 + an)
        inf = 1 / an_
        tau_in = (q10 * self.a0n * an_) / bn
        return self.cnexp(n, inf, tau_in, dt)

    def dldt(self, l, vm, dt: float):
        q10 = self.q10_4()
        al = self.alphal(vm)
        bl = self.betal(vm)
        al_ = (1 + al)
        inf = 1 / al_
        tau_in = (q10 * self.a0l * al_) / bl
        return self.cnexp(l, inf, tau_in, dt)
    
    def expM1(self, x, y):
        b = x / (torch.exp(x/y) - 1)
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                b = torch.where(torch.isnan(b), y*(1 - x/y/2), b)
                b = torch.nan_to_num(b)
        return b
    
    def handle_(self, b):
        if self.handle_nan:
            if torch.any(torch.isnan(b)):
                return torch.nan_to_num(b, nan=1e-6, posinf=1e6, neginf=1e-6)
        return b

    def alpham(self, vm):
        vm = vm + 65 + self.mshift
        return self.am1 * self.expM1(self.am2-vm, self.am3)

    def betam(self, vm):
        vm = vm + 65 + self.mshift
        return self.bm1 * self.expM1(vm-self.bm2, self.bm3)

    def alphah(self, vm):
        vm = vm + 65 + self.hshift
        b = self.ah1 * torch.exp((self.ah2-vm) / self.ah3)
        return self.handle_(b)

    def betah(self, vm):
        vm = vm + 65 + self.hshift
        b = self.bh1 / (torch.exp((self.bh2-vm) / self.bh3) + self.bh4)
        return self.handle_(b)
    
    def alphan(self, vm):
        b = torch.exp(1e-3*self.zetan*(vm-self.vhalfn)*9.648e4/(8.315*(273.16+self.temp)))
        return self.handle_(b)
    
    def betan(self, vm):
        b = torch.exp(1e-3*self.zetan*self.gmn*(vm-self.vhalfn)*9.648e4/(8.315*(273.16+self.temp)))
        return self.handle_(b) 

    def alphal(self, vm):
        b = torch.exp(1e-3*self.zetal*(vm-self.vhalfl)*9.648e4/(8.315*(273.16+self.temp))) 
        return self.handle_(b)

    def betal(self, vm):
        b = torch.exp(1e-3*self.zetal*self.gml*(vm-self.vhalfl)*9.648e4/(8.315*(273.16+self.temp))) 
        return self.handle_(b)
    
    def ionic_currents(self, states: List[Tensor], gbar: List[Tensor]):
        m, h, n, l, vm = states
        gnabar, gkbar, gl = gbar
        i_ion = (
            (gnabar[:, None, None] * m**3 * h * (vm - self.ena))
            + (gkbar[:, None, None] * n**3 * l * (vm - self.ek))
            + (gl[:, None, None] * (vm - self.el))
        )  # mA
        return i_ion

    def prepare_parameters(self, diameters):
        area = self.area(diameters)  # cm2
        cm = self.cm * area  # F
        ra = self.ra(diameters)  # ohm

        gnabar = self.gnabar * area
        gkbar = self.gkbar * area
        gl = self.gl * area

        return cm, ra, (gnabar, gkbar, gl)

    def update_gvs(self, states: List[Tensor], dt: float):
        m, h, n, l, vm = states
        m = self.dmdt(m, vm, dt)
        h = self.dhdt(h, vm, dt)
        n = self.dndt(n, vm, dt)
        l = self.dldt(l, vm, dt)
        return [m, h, n, l, vm]

    def compile(self, nodes=16, axons=1):
        return super().compile(nodes, axons)

    def ic(self) -> Tensor:
        m = 0.00401
        h = 0.98147
        n = 0.00208
        l = 0.57421
        v = -65.0

        dtype = None
        if self.fp32:
            dtype = torch.float

        ic = torch.tensor([m, h, n, l, v], dtype=dtype, device=self.device()).reshape(
            1, -1, 1
        )
        return ic
