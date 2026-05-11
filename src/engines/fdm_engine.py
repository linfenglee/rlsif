from typing import Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from src.engines.base_engine import BaseEngine


class FDMEngine(BaseEngine):
    """"""

    def __init__(self, market_data, model_data, parameter_data):
        super().__init__(market_data, model_data, parameter_data)
        self.dtt = self.mkt.T / (1 - 2 * self.pmt.mu) / self.mdl.N
        self.dte = (self.mdl.eMax - self.mdl.eMin) / self.mdl.I
        self.et = np.linspace(self.mdl.eMin, self.mdl.eMax, self.mdl.I + 1)

    def calculate_vectors(self, n: int) -> Tuple:
        """"""
        tau = self.mdl.T - n * self.mdl.dt
        c2 = 0.5 * np.square(self.pmt.gamma / self.mdl.de) * np.ones(self.mdl.I + 1)
        c1 = -0.5 * self.pmt.mu * self.et / tau / self.mdl.de
        left = -c2 + c1 * (c1 < 0)
        mid = 1 / self.mdl.dt + self.mkt.rf + 2 * c2 + np.abs(c1)
        right = -c2 - c1 * (c1 > 0)
        left[-1], right[0] = 0, 0
        return left, mid, right

    def check_tolerance(self, vn1: np.array, vn2: np.array) -> bool:
        """"""
        iter_error = np.linalg.norm(vn1 - vn2) / np.linalg.norm(vn2)
        if iter_error <= self.mdl.E:
            return True
        else:
            return False

    def calculate_penalty2(self, vw: np.array, uw: np.array) -> Tuple:
        """"""
        pw = np.maximum(np.maximum(vw + self.et - self.mkt.C0, uw - self.et - self.mkt.C0), 0)
        pv = np.maximum(-vw - self.et - self.mkt.C1, 0)
        pu = np.maximum(-uw + self.et - self.mkt.C1, 0)
        fvw = self.mdl.B * (pw - pv)
        fuw = self.mdl.B * (pw - pu)
        dfw1 = self.mdl.B * (pw > 0) * (vw - uw > -2 * self.et)
        dfw2 = self.mdl.B * (pw > 0) * (uw - vw > 2 * self.et)
        dfv = self.mdl.B * (pv > 0)
        dfu = self.mdl.B * (pu > 0)
        return fvw, fuw, dfw1, dfw2, dfv, dfu

    def calculate_penalty3(self, v: np.array, u: np.array, w: np.array) -> Tuple:
        """"""
        pw = np.maximum(np.maximum(np.maximum(v + self.et - self.mkt.C0, u - self.et - self.mkt.C0), 0) - w, 0)
        pv = np.maximum(np.maximum(w - self.et - self.mkt.C1, 0) - v, 0)
        pu = np.maximum(np.maximum(w + self.et - self.mkt.C1, 0) - u, 0)
        fw = self.mdl.B * pw
        fv = self.mdl.B * pv
        fu = self.mdl.B * pu
        dfw = -self.mdl.B * (pw > 0)
        dfv = -self.mdl.B * (pv > 0)
        dfu = -self.mdl.B * (pu > 0)
        return fw, fv, fu, dfw, dfv, dfu

    def penalty_method2(self) -> Tuple:
        """Penalty Method for Long, Short, & Flat PDEs

        Simplified: keep original separate computation but use a small helper
        to build/solve tridiagonal systems to reduce duplicated code.
        """

        I, N = self.mdl.I, self.mdl.N
        vws = np.zeros(shape=(I + 1, N))
        uws = np.zeros(shape=(I + 1, N))
        vw = uw = vw1 = uw1 = -self.mkt.C1 * np.ones(I + 1)

        for n in tqdm(reversed(range(N))):
            left, middle, right = self.calculate_vectors(n)
            while True:
                fvw, fuw, dfw1, dfw2, dfv, dfu = self.calculate_penalty2(vw, uw)

                mid1 = middle + dfw1 + dfv
                mid2 = middle + dfw2 + dfu
                bvw = vw1 / self.mdl.dt - (fvw - (dfw1 + dfv) * vw)
                buw = uw1 / self.mdl.dt - (fuw - (dfw2 + dfu) * uw)

                bvw[0] = mid1[0] * (-self.mdl.eMin - self.mkt.C1)
                bvw[-1] = mid1[-1] * (-self.mdl.eMax + self.mkt.C0)
                buw[0] = mid2[0] * (self.mdl.eMin + self.mkt.C0)
                buw[-1] = mid2[-1] * (self.mdl.eMax - self.mkt.C1)

                vw_new = self._solve_tridiag(mid1, left, right, bvw)
                uw_new = self._solve_tridiag(mid2, left, right, buw)

                tol_cond1 = self.check_tolerance(vw_new, vw)
                tol_cond2 = self.check_tolerance(uw_new, uw)

                vw = vw_new
                uw = uw_new

                if tol_cond1 and tol_cond2:
                    break

            vws[:, n] = vw1 = vw_new
            uws[:, n] = uw1 = uw_new
        return vws, uws

    def penalty_method3(self) -> Tuple:
        """Penalty method 3: compute v (and u,w) using results from method 2.

        This implementation reuses `penalty_method2` to obtain `vw` and `uw`
        per timestep and then solves the single tridiagonal system for `v`.
        """

        I, N = self.mdl.I, self.mdl.N
        vs = np.zeros(shape=(I + 1, N))
        us = np.zeros(shape=(I + 1, N))
        ws = np.zeros(shape=(I + 1, N))
        v = v1 = -self.mkt.C1 * np.ones(I + 1)

        vws, uws = self.penalty_method2()

        for n in tqdm(reversed(range(N))):
            left, middle, right = self.calculate_vectors(n)
            vw = vws[:, n]
            uw = uws[:, n]

            b = v1 / self.mdl.dt + self.mdl.B * np.maximum(-vw - self.et - self.mkt.C1, 0)
            b[0] = middle[0] * (-2 * self.mdl.eMin - self.mkt.C0 - self.mkt.C1)
            b[-1] = 0

            v_new = self._solve_tridiag(middle, left, right, b)
            vs[:, n] = v1 = v_new
            ws[:, n] = v_new - vw
            us[:, n] = uw + v_new - vw

        return vs, us, ws

    def _solve_tridiag(self, mid: np.array, left: np.array, right: np.array, b: np.array) -> np.array:
        """Helper to assemble the tridiagonal matrix and solve the linear system.

        Encapsulates the repeated pattern of building the sparse tridiagonal
        matrix with SciPy and calling spsolve.
        """
        I = self.mdl.I
        mat = diags([left[1:], mid, right[:-1]], offsets=[-1, 0, 1], shape=(I + 1, I + 1), format="csc")
        return spsolve(mat, b)
