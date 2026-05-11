from typing import Any, Tuple

import numpy as np
import torch

from .base import MarketData, ModelData, ParameterData
from .model import ValueNet


DTYPE = torch.float32



class BaseRLEngine(object):
    """
    Basic RL Engine
    """

    def __init__(
            self,
            mkt: MarketData,
            mdl: ModelData,
            pmt: ParameterData
    ):
        """"""
        self.mkt = mkt
        self.mdl = mdl
        self.pmt = pmt

        self.df = np.exp(-self.mkt.rf * self.mdl.dt)

        self.ts = np.linspace(0, self.mdl.T, self.mdl.N + 1)

    def rescale_dv(self, tdv: torch.Tensor) -> torch.Tensor:
        """"""
        dv1 = torch.nn.functional.relu(tdv) * torch.tensor(self.mkt.C0, dtype=DTYPE)
        dv2 = torch.nn.functional.relu(-tdv) * torch.tensor(-self.mkt.C1, dtype=DTYPE)
        return dv1 + dv2

    def H(self, pi: np.ndarray) -> np.ndarray:
        """"""
        return pi - pi * np.log(pi)

    def clamp(self, log_pi: np.ndarray) -> np.ndarray:
        """"""
        clamp_log_pi = np.minimum(log_pi, 10)
        return np.asarray(np.maximum(np.minimum(np.exp(clamp_log_pi), 1 / self.mdl.dt), 1e-6), dtype=float)

    def train_sim(self, ets: np.ndarray) -> float:
        """"""
        raise NotImplementedError

    def train_real(self, ets: np.ndarray) -> float:
        """"""
        raise NotImplementedError

    def calc_boundaries(self, t: float) -> Tuple:
        """"""
        raise NotImplementedError


class RLEngine(BaseRLEngine):
    """Neural RL engine that learns the value functions and policies."""

    def __init__(
        self,
        mkt: MarketData,
        mdl: ModelData,
        pmt: ParameterData,
    ):
        super().__init__(mkt, mdl, pmt)

        self.dv1 = ValueNet(fc_layers=self.mdl.vlf_layers, activation=torch.nn.ReLU(), output_fn=torch.nn.Hardtanh())
        self.dv2 = ValueNet(fc_layers=self.mdl.vsf_layers, activation=torch.nn.ReLU(), output_fn=torch.nn.Hardtanh())

        self.optim1 = torch.optim.Adam(self.dv1.parameters(), lr=self.mdl.lr)
        self.optim2 = torch.optim.Adam(self.dv2.parameters(), lr=self.mdl.lr)

        self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optim1, gamma=self.mdl.decay)
        self.scheduler2 = torch.optim.lr_scheduler.ExponentialLR(self.optim2, gamma=self.mdl.decay)

        self.criterion = torch.nn.MSELoss()

    def calc_dvs(self, st: np.ndarray, training: bool = False) -> Tuple[Any, Any]:
        temp_dv1 = self.dv1.forward(torch.tensor(st, dtype=DTYPE))
        temp_dv2 = self.dv2.forward(torch.tensor(st, dtype=DTYPE))
        dv1 = self.rescale_dv(temp_dv1)
        dv2 = self.rescale_dv(temp_dv2)
        if training:
            return dv1, dv2
        return dv1.detach().numpy(), dv2.detach().numpy()

    def calc_pts(self, st: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        dv1, dv2 = self.calc_dvs(st, training=False)

        log_psf = np.asarray(-(dv2 + self.mkt.C1) / self.mdl.L, dtype=float)
        log_plf = np.asarray(-(dv1 + self.mkt.C1) / self.mdl.L, dtype=float)
        log_pfs = np.asarray((dv2 - self.mkt.C0) / self.mdl.L, dtype=float)
        log_pfl = np.asarray((dv1 - self.mkt.C0) / self.mdl.L, dtype=float)

        psf = self.clamp(log_psf)
        plf = self.clamp(log_plf)
        pfs = self.clamp(log_pfs)
        pfl = self.clamp(log_pfl)

        return psf, plf, pfs, pfl

    def train_sim(self, ets: np.ndarray) -> float:
        ns = np.ones(shape=(ets.shape[0], 1))
        total_loss = 0.0

        for n in reversed(range(self.mdl.N)):
            tm, tp = ns * self.ts[n], ns * self.ts[n + 1]
            etm, etp = ets[:, n].reshape((-1, 1)), ets[:, n + 1].reshape((-1, 1))
            stm, stp = np.concatenate([tm, etm], axis=1), np.concatenate([tp, etp], axis=1)

            vme1, vme2 = self.calc_dvs(stm, training=True)
            x1 = vme1 - torch.tensor(etm, dtype=DTYPE)
            x2 = vme2 + torch.tensor(etm, dtype=DTYPE)

            psf, plf, pfs, pfl = self.calc_pts(stm)

            vpe1, vpe2 = self.calc_dvs(stp, training=False)
            vp1 = vpe1 - etp
            vp2 = vpe2 + etp

            cft = pfs * (-etm - self.mkt.C0) + pfl * (etm - self.mkt.C0)
            ct1 = (plf * (-etm - self.mkt.C0) - cft) * self.mdl.dt
            ct2 = (psf * (etm - self.mkt.C0) - cft) * self.mdl.dt

            rt1 = self.mdl.L * (self.H(plf) - self.H(pfl) - self.H(pfs)) * self.mdl.dt
            rt2 = self.mdl.L * (self.H(psf) - self.H(pfl) - self.H(pfs)) * self.mdl.dt

            dm11 = 1 - (plf + pfl) * self.mdl.dt
            dm12 = -pfs * self.mdl.dt
            dm21 = 1 - (psf + pfs) * self.mdl.dt
            dm22 = -pfl * self.mdl.dt

            if n == self.mdl.N - 1:
                y1 = torch.tensor(ct1 + rt1 - self.df * self.mkt.C1 * (dm11 + dm12), dtype=DTYPE)
                y2 = torch.tensor(ct2 + rt2 - self.df * self.mkt.C1 * (dm21 + dm22), dtype=DTYPE)
            else:
                y1 = torch.tensor(ct1 + rt1 + self.df * (vp1 * dm11 + vp2 * dm12), dtype=DTYPE)
                y2 = torch.tensor(ct2 + rt2 + self.df * (vp2 * dm21 + vp1 * dm22), dtype=DTYPE)

            loss1 = self.criterion(x1, y1)
            loss2 = self.criterion(x2, y2)

            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()

            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()

            total_loss += loss1.detach().numpy() + loss2.detach().numpy()

        self.scheduler1.step()
        self.scheduler2.step()

        return total_loss

    def train_real(self, ets: np.ndarray) -> float:
        total_loss = 0.0
        length = ets.shape[0]

        for n in reversed(range(1, length, 1)):
            stm, etm = ets[n, :, :], ets[n, :, 1].reshape(-1, 1)

            vme1, vme2 = self.calc_dvs(stm, training=True)
            x1 = vme1 - torch.tensor(etm, dtype=DTYPE)
            x2 = vme2 + torch.tensor(etm, dtype=DTYPE)

            psf, plf, pfs, pfl = self.calc_pts(stm)

            cft = pfs * (-etm - self.mkt.C0) + pfl * (etm - self.mkt.C0)
            ct1 = (plf * (-etm - self.mkt.C0) - cft) * self.mdl.dt
            ct2 = (psf * (etm - self.mkt.C0) - cft) * self.mdl.dt

            rt1 = self.mdl.L * (self.H(plf) - self.H(pfl) - self.H(pfs)) * self.mdl.dt
            rt2 = self.mdl.L * (self.H(psf) - self.H(pfl) - self.H(pfs)) * self.mdl.dt

            dm11 = 1 - (plf + pfl) * self.mdl.dt
            dm12 = -pfs * self.mdl.dt
            dm21 = 1 - (psf + pfs) * self.mdl.dt
            dm22 = -pfl * self.mdl.dt

            if n == length - 1:
                y1 = torch.tensor(ct1 + rt1 - self.df * self.mkt.C1 * (dm11 + dm12), dtype=DTYPE)
                y2 = torch.tensor(ct2 + rt2 - self.df * self.mkt.C1 * (dm21 + dm22), dtype=DTYPE)
            else:
                stp, etp = ets[n, :, :], ets[n, :, 1].reshape(-1, 1)
                vpe1, vpe2 = self.calc_dvs(stp, training=False)
                vp1 = vpe1 - etp
                vp2 = vpe2 + etp
                y1 = torch.tensor(ct1 + rt1 + self.df * (vp1 * dm11 + vp2 * dm12), dtype=DTYPE)
                y2 = torch.tensor(ct2 + rt2 + self.df * (vp2 * dm21 + vp1 * dm22), dtype=DTYPE)

            loss1 = self.criterion(x1, y1)
            loss2 = self.criterion(x2, y2)

            self.optim1.zero_grad()
            loss1.backward()
            self.optim1.step()

            self.optim2.zero_grad()
            loss2.backward()
            self.optim2.step()

            total_loss += loss1.detach().numpy() + loss2.detach().numpy()

        return total_loss

    def calc_boundaries(self, t: float) -> Tuple[float, float, float, float]:
        ts = t * np.ones(shape=[self.mdl.I + 1, 1])
        es = np.linspace(self.mdl.eMin, self.mdl.eMax, self.mdl.I + 1)
        st = np.concatenate([ts, es.reshape(-1, 1)], axis=1)
        dv1, dv2 = self.calc_dvs(st, training=False)

        dv1 = np.squeeze(dv1, axis=-1)
        dv2 = np.squeeze(dv2, axis=-1)

        eil = self.mdl.eMax - (dv1 >= self.mkt.C0).sum() * self.mdl.de
        eis = self.mdl.eMin + (dv2 >= self.mkt.C0).sum() * self.mdl.de
        ecl = self.mdl.eMin + (dv1 <= -self.mkt.C1).sum() * self.mdl.de
        ecs = self.mdl.eMax - (dv2 <= -self.mkt.C1).sum() * self.mdl.de

        return eil, eis, ecl, ecs


__all__ = ["BaseRLEngine", "RLEngine"]


