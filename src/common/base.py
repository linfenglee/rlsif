"""Shared data containers and enums used by both RL and FDM packages."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


@dataclass
class MarketData:
    """Market-level parameters shared by RL and FDM."""

    e0: float = 0.0
    rf: float = 0.07
    T: float = 1.0
    C0: float = 0.50
    C1: float = 0.50


@dataclass
class BaseModelData:
    """Core model parameters shared by RL and FDM."""

    T: float = 1.0
    N: int = 1000
    I: int = 2000
    eMax: float = 10.0
    eMin: float = -10.0
    L: float = 1e-3

    def __post_init__(self) -> None:
        self.dt: float = self.T / self.N
        self.de: float = (self.eMax - self.eMin) / self.I


@dataclass
class ParameterData:
    """Shared stochastic parameters."""

    mu: float = 0.3
    gamma: float = 0.6


@dataclass
class FDMModelData(BaseModelData):
    """FDM model parameters."""

    xMax: float = 10.0
    xMin: float = -10.0

    B: float = 1e5
    E: float = 1e-8


@dataclass
class RLModelData(BaseModelData):
    """RL model parameters built on the shared core model fields."""

    P: int = 1024

    lr: float = 0.01
    decay: float = 0.999

    vlf_layers: Tuple = (2, 21, 21, 1)
    vsf_layers: Tuple = (2, 21, 21, 1)


class BoundaryData:
    """Boundary values used by RL utilities."""

    ILR: float = 1.72
    ISR: float = -1.72
    CLR: float = -0.31
    CSR: float = 0.31


__all__ = ["MarketData", "BaseModelData", "FDMModelData", "RLModelData", "ParameterData"]

