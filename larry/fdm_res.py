from typing import Tuple

import numpy as np
import pandas as pd

from src.common.base import IterAlgo, MarketData, FDMModelData, Mod, ParameterData
from src.fdm.engine import Engine

import matplotlib.pyplot as plt


STEPS = 15


def run_fdm_lf(mkt: MarketData, mdl: FDMModelData, pmt: ParameterData):
    """"""

    fdm_engine = Engine(mkt, mdl, pmt)

    vns = fdm_engine.penalty_method1()
    # vns = fdm_engine.implicit_finite_difference_method_reg1()

    if mdl.M == Mod.LF:
        vr = vns[:, 0] + fdm_engine.et
        ilr = mdl.eMax - (vr >= mkt.C0).sum() * mdl.de
        clr = mdl.eMin + (vr <= -mkt.C1).sum() * mdl.de
        print("\n")
        print(f"ILR: {round(ilr, 3)} \t|\t CLR: {round(clr, 3)}")
    else:
        vr = vns[:, 0] - fdm_engine.et
        isr = mdl.eMin + (vr <= mkt.C0).sum() * mdl.de
        csr = mdl.eMax - (vr >= -mkt.C1).sum() * mdl.de
        print("\n")
        print(f"ISR: {round(isr, 3)} \t|\t CSR: {round(csr, 3)}")

    fig, ax = plt.subplots()

    ax.plot(fdm_engine.et, vr, 'b-')
    ax.plot(fdm_engine.et, mkt.C0 * np.ones(len(vr)), 'k--')
    ax.plot(fdm_engine.et, -mkt.C1 * np.ones(len(vr)), 'k--')
    ax.set_xlabel("$e_t$")
    ax.set_ylabel("$v_t$")
    if mdl.M == Mod.LF:
        ax.set_title("$V - W + e_t$")
    else:
        ax.set_title("$U - W - e_t$")

    fig.show()

    return vns


def run_fdm_lfs(mkt: MarketData, mdl: FDMModelData, pmt: ParameterData) -> Tuple:
    """"""
    fdm_engine = Engine(mkt, mdl, pmt)

    vws, uws = fdm_engine.penalty_method2()
    # vns = fdm_engine.implicit_finite_difference_method_reg2()

    fig, ax = plt.subplots(figsize=(8, 6))

    lr = vws[:, 0] + fdm_engine.et
    sr = uws[:, 0] - fdm_engine.et

    ilr = fdm_engine.mdl.eMax - (lr >= fdm_engine.mkt.C0).sum() * fdm_engine.mdl.de
    isr = fdm_engine.mdl.eMin + (sr >= fdm_engine.mkt.C0).sum() * fdm_engine.mdl.de
    clr = fdm_engine.mdl.eMin + (lr <= -fdm_engine.mkt.C1).sum() * fdm_engine.mdl.de
    csr = fdm_engine.mdl.eMax - (sr <= -fdm_engine.mkt.C1).sum() * fdm_engine.mdl.de

    print("\n")
    print(f"ILR: {round(ilr, 3)} \t|\t CLR: {round(clr, 3)}")
    print(f"ISR: {round(isr, 3)} \t|\t CSR: {round(csr, 3)}")

    ax.plot(fdm_engine.et, lr, 'b-', label="$U^l(\\varepsilon, t) - U^f(\\varepsilon, t) + \\varepsilon_t$")
    ax.plot(fdm_engine.et, sr, 'r-', label="$U^s(\\varepsilon, t) - U^f(\\varepsilon, t) - \\varepsilon_t$")
    ax.plot(fdm_engine.et, fdm_engine.mkt.C0 * np.ones(len(lr)), 'k--')
    ax.plot(fdm_engine.et, -fdm_engine.mkt.C1 * np.ones(len(lr)), 'k--')
    ax.set_xlabel("$\\varepsilon_t$")
    ax.set_ylabel("$value$")
    ax.legend()
    ax.set_title("Optimal Boundaries ($T = 1$ and $t = 0$)")

    fig.show()

    # fig.savefig("boundaries.png")
    fig.savefig("boundaries.eps", format="eps")

    return vws, uws


def run_fdm_lfs_all(mkt: MarketData, mdl: FDMModelData, pmt: ParameterData) -> Tuple:
    """"""
    fdm_engine = Engine(mkt, mdl, pmt)

    vs, us, ws = fdm_engine.penalty_method3()
    # vs, us, ws = fdm_engine.penalty_method4()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fdm_engine.et[STEPS:-STEPS], vs[STEPS:-STEPS, 0], 'b-', label="$U^{l}(\\varepsilon, t)$")
    ax.plot(fdm_engine.et[STEPS:-STEPS], us[STEPS:-STEPS, 0], 'r-', label="$U^{s}(\\varepsilon, t)$")
    ax.plot(fdm_engine.et[STEPS:-STEPS], ws[STEPS:-STEPS, 0], 'g-', label="$U^{f}(\\varepsilon, t)$")
    ax.set_xlabel("$\\varepsilon_t$")
    ax.set_ylabel("$value$")
    ax.legend()
    ax.set_title("Value Functions ($T = 1$ and $t = 0$)")

    fig.show()

    # fig.savefig("values.png")
    fig.savefig("values.eps", format="eps")

    return vs, us, ws


def run_fdm_boundaries(mkt: MarketData, mdl: FDMModelData, pmt: ParameterData) -> pd.DataFrame:
    """"""
    fdm_engine = Engine(mkt, mdl, pmt)

    vws, uws = fdm_engine.penalty_method2()

    df = pd.DataFrame(columns=["ILR", "ISR", "CLR", "CSR"])
    for i in range(vws.shape[1]):
        lr = vws[:, i] + fdm_engine.et
        sr = uws[:, i] - fdm_engine.et

        ilr = fdm_engine.mdl.eMax - (lr >= fdm_engine.mkt.C0).sum() * fdm_engine.mdl.de
        isr = fdm_engine.mdl.eMin + (sr >= fdm_engine.mkt.C0).sum() * fdm_engine.mdl.de
        clr = fdm_engine.mdl.eMin + (lr <= -fdm_engine.mkt.C1).sum() * fdm_engine.mdl.de
        csr = fdm_engine.mdl.eMax - (sr <= -fdm_engine.mkt.C1).sum() * fdm_engine.mdl.de

        df.loc[i * fdm_engine.mdl.dt] = [ilr, isr, clr, csr]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df.index, df["ILR"], "b-", label="ILR")
    ax.plot(df.index, df["ISR"], "r-", label="ISR")
    ax.plot(df.index, df["CLR"], "c-", label="CLR")
    ax.plot(df.index, df["CSR"], "m-", label="CSR")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\varepsilon$")
    ax.legend()
    ax.set_title("Optimal Switching Boundaries")
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=6.5)

    fig.show()

    # fig.savefig("fdm_boundaries.png")
    fig.savefig("fdm_boundaries.png")

    return df


if __name__ == "__main__":

    # 2018-2021: 0.0019  0.679
    # 2014-2018: 0.0011  1.340
    # 2010-2014: 0.0022  0.561
    # 2010-2021: 0.0014  0.955

    market_data = MarketData(rf=0.07, T=1.0, C0=3.0, C1=3.0)
    model_data = FDMModelData(T=1.0, L=1e-2, A=IterAlgo.Newton, M=Mod.LSF)

    param_list = []

    parameter_data = ParameterData()
    parameter_data.mu = 0.0022
    parameter_data.gamma = 0.561
    param_list.append(("2010_2014", parameter_data))

    parameter_data = ParameterData()
    parameter_data.mu = 0.0011
    parameter_data.gamma = 1.340
    param_list.append(("2014_2018", parameter_data))

    parameter_data = ParameterData()
    parameter_data.mu = 0.0019
    parameter_data.gamma = 0.679
    param_list.append(("2018_2021", parameter_data))

    parameter_data = ParameterData()
    parameter_data.mu = 0.0014
    parameter_data.gamma = 0.955
    param_list.append(("2010_2021", parameter_data))

    for nm, parameter_data in param_list:
        print(f"Running FDM for {nm} parameters: mu={parameter_data.mu}, gamma={parameter_data.gamma}")
        res = run_fdm_boundaries(market_data, model_data, parameter_data)
        # res.to_csv(f"res_fdm_boundaries_{nm}.csv")
        # print(f"Results saved to res_fdm_boundaries_{nm}.csv\n")

    # # parameter_data = ParameterData(mu=0.0019, gamma=0.679)
    # parameter_data = ParameterData(mu=0.0014, gamma=0.955)
    # # parameter_data = ParameterData(mu=0.30, gamma=0.60)
    #
    # # res = test_fdm_lf(market_data, model_data, parameter_data)
    #
    # # res1, res2 = test_fdm_lfs(market_data, model_data, parameter_data)
    # #
    # # res1, res2, res3 = test_fdm_lfs_all(market_data, model_data, parameter_data)
    #
    # res = run_fdm_boundaries(market_data, model_data, parameter_data)
    #
    # res.to_csv("res_fdm_boundaries_2010_2021.csv")