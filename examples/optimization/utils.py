import numpy as np
import torch

from cajal.opt.loss import PredictionLoss


# activation statistics


def percent_on_target_active(active, target, weights):
    target_active_area = np.sum(np.logical_and(active, target) * weights)
    total_target_area = np.sum(target * weights)
    return 100 * target_active_area / total_target_area


def percent_off_target_active(active, target, weights):
    off_target_active_area = np.sum(
        np.logical_and(active, np.logical_not(target)) * weights
    )
    total_off_target_area = np.sum(np.logical_not(target) * weights)
    return 100 * off_target_active_area / total_off_target_area


# initial condition generator


def ic(n_axons, nodes):
    m_init = 0.0732093
    h_init = 0.62069505
    p_init = 0.20260409
    s_init = 0.04302994
    v_init = -80.0

    ic = torch.tensor([m_init, h_init, p_init, s_init, v_init]).reshape(1, -1, 1)
    ic = ic.repeat(n_axons, 1, nodes)

    return ic.double().cuda()


# internodal length calculator for original MRG


def deltax(diam):
    return -8.215284e00 * diam**2 + 2.724201e02 * diam + -7.802411e02


# -- loss functions --


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
