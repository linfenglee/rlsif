from typing import Tuple

import torch

from ..common.base import Mod

DTYPE = torch.float32


class BoundaryData:
    """Boundary values used by RL utilities."""

    ILR: float = 1.72
    ISR: float = -1.72
    CLR: float = -0.31
    CSR: float = 0.31


class ModelData:
    """RL model parameters built on the shared core model fields."""

    T: float = 1.0
    N: int = 50
    I: int = 1000
    P: int = 1024

    eMax: float = 10.0
    eMin: float = -10.0

    L: float = 1e-3

    lr: float = 0.01
    decay: float = 0.999

    M: Mod = Mod.LF

    vlf_layers: Tuple = (2, 21, 21, 1)
    vsf_layers: Tuple = (2, 21, 21, 1)

    def __init__(
        self,
        T: float = 1.0,
        N: int = 50,
        I: int = 1000,
        P: int = 1024,
        eMax: float = 10.0,
        eMin: float = -10.0,
        L: float = 1e-3,
        lr: float = 0.01,
        decay: float = 0.999,
        M: Mod = Mod.LF,
        vlf_layers: Tuple = (2, 21, 21, 1),
        vsf_layers: Tuple = (2, 21, 21, 1),
    ) -> None:
        self.T = T
        self.N = N
        self.I = I
        self.P = P
        self.eMax = eMax
        self.eMin = eMin
        self.L = L
        self.lr = lr
        self.decay = decay
        self.M = M
        self.vlf_layers = vlf_layers
        self.vsf_layers = vsf_layers
        self.dt: float = self.T / self.N
        self.de: float = (self.eMax - self.eMin) / self.I


class ParameterData:
    """Shared stochastic parameters for the RL package."""

    mu: float = 0.3
    gamma: float = 0.6

    def __init__(self, mu: float = 0.3, gamma: float = 0.6) -> None:
        self.mu = mu
        self.gamma = gamma


__all__ = ["DTYPE", "BoundaryData", "Mod", "MarketData", "ModelData", "ParameterData"]
