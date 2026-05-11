from typing import Any, Callable, Optional, Tuple

import logging
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from tqdm import tqdm

from src.common.base import IterAlgo, MarketData, Mod, FDMModelData, ParameterData


logger = logging.getLogger(__name__)

StateTuple = Tuple[Any, ...]


class FDMError(RuntimeError):
    """Custom error for FDM-related numerical issues."""


class Engine(object):
    """Finite-difference solver engine.

    Provides several penalty and implicit schemes. This class is a thin
    refactor of the original prototype with clearer typing and logging.
    """

    def __init__(
        self,
        market_data: MarketData,
        model_data: FDMModelData,
        parameter_data: ParameterData,
        *,
        verbose: bool = False,
    ) -> None:
        """Create a solver instance.

        Args:
            market_data: market parameters
            model_data: discretization and model parameters
            parameter_data: model-specific parameters
            verbose: enable progress output (tqdm)
        """

        self.mkt = market_data
        self.mdl = model_data
        self.pmt = parameter_data
        self.verbose = verbose

        # time-step in transformed coordinate and epsilon step
        self.dtt = self.mkt.T / (1 - 2 * self.pmt.mu) / self.mdl.N
        self.dte = (self.mdl.eMax - self.mdl.eMin) / self.mdl.I

        self.et: np.ndarray = np.linspace(self.mdl.eMin, self.mdl.eMax, self.mdl.I + 1)

    def _build_tridiagonal_matrix(self, left: np.ndarray, mid: np.ndarray, right: np.ndarray):
        return diags(
            [left[1:], mid, right[:-1]],
            offsets=[-1, 0, 1],
            shape=(self.mdl.I + 1, self.mdl.I + 1),
            format="csc",
        )

    def _build_linear_system(self, previous, current, middle, residual, derivative) -> Tuple[np.ndarray, np.ndarray]:
        previous = np.asarray(previous, dtype=float)
        current = np.asarray(current, dtype=float)
        middle = np.asarray(middle, dtype=float)
        residual = np.asarray(residual, dtype=float)
        derivative = np.asarray(derivative, dtype=float)
        if self.mdl.A == IterAlgo.Simple:
            return np.asarray(middle, dtype=float), np.asarray(previous / self.mdl.dt - residual, dtype=float)

        mid = np.asarray(middle + derivative, dtype=float)
        rhs = np.asarray(previous / self.mdl.dt - (residual - derivative * current), dtype=float)
        return mid, rhs

    def _set_boundary_values(self, rhs: np.ndarray, mid: np.ndarray, left_value: float, right_value: float) -> None:
        rhs[0] = mid[0] * left_value
        rhs[-1] = mid[-1] * right_value

    def _solve_tridiagonal(self, left: np.ndarray, mid: np.ndarray, right: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return spsolve(self._build_tridiagonal_matrix(left, mid, right), rhs)

    def _iterate_until_converged(self, state: StateTuple, step: Callable[[StateTuple], StateTuple]) -> StateTuple:
        current = tuple(np.array(item, copy=True) for item in state)
        while True:
            updated = step(current)
            if all(self.check_tolerance(new, old) for new, old in zip(updated, current)):
                return updated
            current = updated

    def calculate_vectors(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the tridiagonal coefficients (left, mid, right) for time-step n.

        The three arrays correspond to the lower diagonal, main diagonal and
        upper diagonal entries of the discretized linear system.
        """

        tau = self.mdl.T - n * self.mdl.dt
        c2 = 0.5 * np.square(self.pmt.gamma / self.mdl.de) * np.ones(self.mdl.I + 1)
        c1 = -0.5 * self.pmt.mu * self.et / tau / self.mdl.de
        left = -c2 + c1 * (c1 < 0)
        mid = 1 / self.mdl.dt + self.mkt.rf + 2 * c2 + np.abs(c1)
        right = -c2 - c1 * (c1 > 0)
        left[-1], right[0] = 0, 0
        return left, mid, right

    def calculate_transform_vectors(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tridiagonal coefficients for the transformed-time scheme."""

        tau = n * self.dtt
        c0 = self.mkt.rf * np.power((1 - 2 * self.pmt.mu) * tau, 2 * self.pmt.mu / (1 - 2 * self.pmt.mu))
        c2 = 0.5 * np.square(self.pmt.gamma / self.dte)
        left = np.full(self.mdl.I + 1, -c2)
        mid = np.full(self.mdl.I + 1, 1 / self.dtt + 2 * c2 + c0)
        right = np.full(self.mdl.I + 1, -c2)
        left[-1], right[0] = 0, 0
        return left, mid, right

    def check_tolerance(self, vn1: np.ndarray, vn2: np.ndarray, *, tol: Optional[float] = None) -> bool:
        """Check iterative tolerance safely.

        Uses relative error but guards against near-zero denominators by
        falling back to absolute error when appropriate.
        """

        if tol is None:
            tol = self.mdl.E

        denom = np.linalg.norm(vn2)
        num = np.linalg.norm(vn1 - vn2)
        if denom < 1e-12:
            # fallback to absolute error
            return bool(num <= tol)
        return bool((num / denom) <= tol)

    def calculate_penalty1(self, vn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""
        if self.mdl.M == Mod.LF:
            pw = np.maximum(vn + self.et - self.mkt.C0, 0)
            pv = np.maximum(-vn - self.et - self.mkt.C1, 0)
        else:
            pw = np.maximum(vn - self.et - self.mkt.C0, 0.0)
            pv = np.maximum(-vn + self.et - self.mkt.C1, 0.0)
        fv = np.asarray(self.mdl.B * (pw - pv), dtype=float)
        dff = np.asarray(self.mdl.B * (pw > 0).astype(float), dtype=float)
        dfl = np.asarray(self.mdl.B * (pv > 0).astype(float), dtype=float)
        return fv, dff, dfl

    def calculate_penalty2(self, vw: np.ndarray, uw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """"""
        pw = np.maximum(np.maximum(vw + self.et - self.mkt.C0, uw - self.et - self.mkt.C0), 0)
        pv = np.maximum(-vw - self.et - self.mkt.C1, 0)
        pu = np.maximum(-uw + self.et - self.mkt.C1, 0)
        fvw = np.asarray(self.mdl.B * (pw - pv), dtype=float)
        fuw = np.asarray(self.mdl.B * (pw - pu), dtype=float)
        dfw1 = np.asarray(self.mdl.B * (pw > 0).astype(float) * (vw - uw > -2 * self.et).astype(float), dtype=float)
        dfw2 = np.asarray(self.mdl.B * (pw > 0).astype(float) * (uw - vw > 2 * self.et).astype(float), dtype=float)
        dfv = np.asarray(self.mdl.B * (pv > 0).astype(float), dtype=float)
        dfu = np.asarray(self.mdl.B * (pu > 0).astype(float), dtype=float)
        return fvw, fuw, dfw1, dfw2, dfv, dfu

    def calculate_penalty3(self, v: np.ndarray, u: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """"""
        pw = np.maximum(np.maximum(np.maximum(v + self.et - self.mkt.C0, u - self.et - self.mkt.C0), 0) - w, 0)
        pv = np.maximum(np.maximum(w - self.et - self.mkt.C1, 0) - v, 0)
        pu = np.maximum(np.maximum(w + self.et - self.mkt.C1, 0) - u, 0)
        fw = np.asarray(self.mdl.B * pw, dtype=float)
        fv = np.asarray(self.mdl.B * pv, dtype=float)
        fu = np.asarray(self.mdl.B * pu, dtype=float)
        dfw = np.asarray(-self.mdl.B * (pw > 0).astype(float), dtype=float)
        dfv = np.asarray(-self.mdl.B * (pv > 0).astype(float), dtype=float)
        dfu = np.asarray(-self.mdl.B * (pu > 0).astype(float), dtype=float)
        return fw, fv, fu, dfw, dfv, dfu

    def calculate_transform_penalty(self, vn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Placeholder for transformed penalty calculation (not implemented)."""

        raise NotImplementedError("calculate_transform_penalty is not implemented")

    def calculate_regularization1(self, vnk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        r0 = np.exp((vnk + self.et - self.mkt.C0) / self.mdl.L)
        r1 = np.exp(-(vnk + self.et + self.mkt.C1) / self.mdl.L)
        r = self.mdl.L * (r0 - r1)
        dr = r0 + r1
        return r, dr

    def calculate_regularization2(self, vwk: np.ndarray, uwk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Regularization for paired variables (not implemented)."""

        raise NotImplementedError("calculate_regularization2 is not implemented")

    def show_progress(self, n: int) -> None:
        """Log a short textual progress indicator.

        Kept for backward compatibility with the prototype; modern callers
        should prefer `tqdm` or the ``verbose`` flag on the engine.
        """

        p = int(100 * (1 - n / self.mdl.N) / 2)
        logger.info("Progress: [%-50s]%.2f%%", "#" * p, 2 * p)

    def projected_successive_over_relaxation(self):
        """PSOR solver (not implemented)."""

        raise NotImplementedError("projected_successive_over_relaxation is not implemented")

    def implicit_finite_difference_method(self):
        """Placeholder for the implicit FDM entry point (not implemented)."""

        raise NotImplementedError("implicit_finite_difference_method is not implemented")

    def penalty_method1(self) -> np.ndarray:
        """Penalty method for the long/flat or short/flat PDE."""

        vns = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        vn1 = -self.mkt.C1 * np.ones(self.mdl.I + 1)
        for n in tqdm(reversed(range(self.mdl.N)), disable=not self.verbose):
            left, middle, right = self.calculate_vectors(n)
            boundary_values = (
                (-self.mdl.eMin - self.mkt.C1, -self.mdl.eMax + self.mkt.C0)
                if self.mdl.M == Mod.LF
                else (self.mdl.eMin + self.mkt.C0, self.mdl.eMax - self.mkt.C1)
            )

            def step(state: StateTuple) -> StateTuple:
                (vn,) = state
                fv, dff, dfl = self.calculate_penalty1(vn)
                mid, rhs = self._build_linear_system(vn1, vn, middle, fv, dff + dfl)
                self._set_boundary_values(rhs, mid, *boundary_values)
                return (self._solve_tridiagonal(left, mid, right, rhs),)

            (vn_new,) = self._iterate_until_converged((vn1,), step)

            vns[:, n] = vn1 = vn_new
        return vns

    def penalty_method2(self) -> Tuple[np.ndarray, np.ndarray]:
        """Penalty method for the long, short, and flat PDE system."""

        vws = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        uws = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        vw1 = uw1 = -self.mkt.C1 * np.ones(self.mdl.I + 1)
        for n in tqdm(reversed(range(self.mdl.N)), disable=not self.verbose):
            left, middle, right = self.calculate_vectors(n)
            def step(state: StateTuple) -> StateTuple:
                vw, uw = state
                fvw, fuw, dfw1, dfw2, dfv, dfu = self.calculate_penalty2(vw, uw)

                mid1, bvw = self._build_linear_system(vw1, vw, middle, fvw, dfw1 + dfv)
                mid2, buw = self._build_linear_system(uw1, uw, middle, fuw, dfw2 + dfu)

                self._set_boundary_values(bvw, mid1, -self.mdl.eMin - self.mkt.C1, -self.mdl.eMax + self.mkt.C0)
                self._set_boundary_values(buw, mid2, self.mdl.eMin + self.mkt.C0, self.mdl.eMax - self.mkt.C1)

                return (
                    self._solve_tridiagonal(left, mid1, right, bvw),
                    self._solve_tridiagonal(left, mid2, right, buw),
                )

            vw_new, uw_new = self._iterate_until_converged((vw1, uw1), step)

            vws[:, n] = vw1 = vw_new
            uws[:, n] = uw1 = uw_new
        return vws, uws

    def penalty_method3(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Penalty method for the three-state system built on top of method 2."""

        vs = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        us = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        ws = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        v1 = -self.mkt.C1 * np.ones(self.mdl.I + 1)
        vws, uws = self.penalty_method2()
        for n in tqdm(reversed(range(self.mdl.N)), disable=not self.verbose):
            left, middle, right = self.calculate_vectors(n)
            vw = vws[:, n]
            uw = uws[:, n]
            mat = self._build_tridiagonal_matrix(left, middle, right)

            def step(state: StateTuple) -> StateTuple:
                (v,) = state
                rhs = v1 / self.mdl.dt + self.mdl.B * np.maximum(-vw - self.et - self.mkt.C1, 0)
                self._set_boundary_values(rhs, middle, -2 * self.mdl.eMin - self.mkt.C0 - self.mkt.C1, 0.0)
                return (spsolve(mat, rhs),)

            (v_new,) = self._iterate_until_converged((v1,), step)

            vs[:, n] = v1 = v_new
            ws[:, n] = v_new - vw
            us[:, n] = uw + v_new - vw
        return vs, us, ws

    def penalty_method4(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Penalty method for the fully coupled three-state PDE system."""

        vs = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        us = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        ws = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        w1 = np.zeros(self.mdl.I + 1) + 1e-9
        v1 = u1 = -self.mkt.C1 * np.ones(self.mdl.I + 1)
        for n in tqdm(reversed(range(self.mdl.N)), disable=not self.verbose):
            left, middle, right = self.calculate_vectors(n)
            def step(state: StateTuple) -> StateTuple:
                v, u, w = state
                fw, fv, fu, dfw, dfv, dfu = self.calculate_penalty3(v, u, w)

                mid1, bv = self._build_linear_system(v1, v, middle, fv, dfv)
                mid2, bu = self._build_linear_system(u1, u, middle, fu, dfu)
                mid3, bw = self._build_linear_system(w1, w, middle, fw, dfw)

                self._set_boundary_values(bv, mid1, -2 * self.mdl.eMin - self.mkt.C0 - self.mkt.C1, 0.0)
                self._set_boundary_values(bu, mid2, 0.0, 2 * self.mdl.eMax - self.mkt.C0 - self.mkt.C1)
                self._set_boundary_values(bw, mid3, -self.mdl.eMin - self.mkt.C0, self.mdl.eMax - self.mkt.C0)

                return (
                    self._solve_tridiagonal(left, mid1, right, bv),
                    self._solve_tridiagonal(left, mid2, right, bu),
                    self._solve_tridiagonal(left, mid3, right, bw),
                )

            v_new, u_new, w_new = self._iterate_until_converged((v1, u1, w1), step)

            vs[:, n] = v1 = v_new
            us[:, n] = u1 = u_new
            ws[:, n] = w1 = w_new
        return vs, us, ws

    def implicit_finite_difference_method_reg1(self) -> np.ndarray:
        """Implicit FDM with regularization."""

        vns = np.zeros(shape=(self.mdl.I + 1, self.mdl.N))
        vn1 = -self.mkt.C1 * np.ones(self.mdl.I + 1)
        for n in tqdm(reversed(range(self.mdl.N)), disable=not self.verbose):
            left, middle, right = self.calculate_vectors(n)
            def step(state: StateTuple) -> StateTuple:
                (vn,) = state
                r, dr = self.calculate_regularization1(vn)
                mid, rhs = self._build_linear_system(vn1, vn, middle, r, dr)
                self._set_boundary_values(rhs, mid, -(self.mdl.eMin + self.mkt.C1), -(self.mdl.eMax - self.mkt.C0))
                return (self._solve_tridiagonal(left, mid, right, rhs),)

            (vn_new,) = self._iterate_until_converged((vn1,), step)

            vns[:, n] = vn1 = vn_new
        return vns

    def run(self):
        """"""

        # wvu = self.penalty_method()

        vns = self.implicit_finite_difference_method_reg1()
        return vns


