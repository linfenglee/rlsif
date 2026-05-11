import time
from typing import Any, List, Protocol, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch

from ..rl import BoundaryData, DTYPE, Engine, Engine2, MarketData, ModelData, Mod, ParameterData

try:
    from dl.feeder.csv_feeder import csv_feeder, bn_feeder, cn_feeder
except ImportError:  # pragma: no cover - optional external dependency
    csv_feeder = bn_feeder = cn_feeder = None

COLUMN_NAME1 = ["Iter", "Loss", "IB", "CB", "Time", "LR"]
COLUMN_NAME2 = ["Iter", "Loss", "IL", "IS", "CL", "CS", "Time", "LR"]


class _GridModel(Protocol):
    T: float
    N: int
    P: int
    dt: float
    eMax: float


class _MarketModel(_GridModel, Protocol):
    e0: float


class _ParamModel(Protocol):
    mu: float
    gamma: float


def sim1(mdl: _GridModel, pmt: _ParamModel):
    """"""
    p = int(getattr(mdl, "P", 1))
    tau = mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)
    zt = np.insert(
        np.cumsum(np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N)), axis=1),
        0, 0, axis=1
    )
    et = pmt.gamma * np.power(tau, 1.0 - pmt.mu) * zt / (2 * pmt.mu - 1)
    return et


def sim2(mdl: _GridModel, pmt: _ParamModel):
    """"""
    p = int(getattr(mdl, "P", 1))
    et = np.zeros(shape=(p, mdl.N + 1))
    at: np.ndarray = np.asarray(pmt.mu / (mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)[:-1]), dtype=float)
    zt = np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N))
    for i, ai in enumerate(at):
        et[:, i + 1] = et[:, i] * (1.0 - ai * mdl.dt) + pmt.gamma * zt[:, i]
    return et


def sim3(mkt: _MarketModel, mdl: _GridModel, pmt: _ParamModel):
    """"""
    p = int(getattr(mdl, "P", 1))
    tau = mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)
    zt = np.insert(
        np.cumsum(np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N)), axis=1),
        0, 0, axis=1
    )
    et = float(getattr(mkt, "e0", 0.0)) * np.power(tau, pmt.mu) / mdl.T + pmt.gamma * np.power(tau, 1.0 - pmt.mu) * zt / (2 * pmt.mu - 1)
    return et


def sim4(mdl: _GridModel, pmt: _ParamModel):
    p = int(getattr(mdl, "P", 1))
    tau = mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)
    zt = np.insert(
        np.cumsum(np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N)), axis=1),
        0, 0, axis=1
    )
    # e0 = np.random.normal(loc=0.0, scale=pmt.gamma * np.sqrt(mdl.T), size=(mdl.P, 1))
    e0 = 2 * (np.random.random(size=(p, 1)) - 0.5) * mdl.eMax
    et = e0 * np.power(tau, pmt.mu) / mdl.T + pmt.gamma * np.power(tau, 1.0 - pmt.mu) * zt / (2 * pmt.mu - 1)
    return et


def sim5(mdl: _GridModel, pmt: _ParamModel, scaler: float = 1.0):
    p = int(getattr(mdl, "P", 1))
    et = np.zeros(shape=(p, mdl.N + 1))
    # et[:, 0] = 0.4 * (np.random.random(size=mdl.P) - 0.5) * mdl.eMax
    et[:, 0] = np.random.normal(loc=0.0, scale=scaler * np.sqrt(mdl.T), size=p)
    at: np.ndarray = np.asarray(pmt.mu / (mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)[:-1]), dtype=float)
    zt = np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N - 1))
    for i, ai in enumerate(at):
        et[:, i + 1] = et[:, i] * (1.0 - ai * mdl.dt) + pmt.gamma * zt[:, i]
    return et


def sim6(mkt: _MarketModel, mdl: _GridModel, pmt: _ParamModel):
    p = int(getattr(mdl, "P", 1))
    et = np.zeros(shape=(p, mdl.N + 1))
    et[:, 0] = float(getattr(mkt, "e0", 0.0))
    at: np.ndarray = np.asarray(pmt.mu / (mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)[:-1]), dtype=float)
    zt = np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N - 1))
    for i, ai in enumerate(at):
        et[:, i + 1] = et[:, i] * (1.0 - ai * mdl.dt) + pmt.gamma * zt[:, i]
    return et


def sim7(mdl: _GridModel, pmt: _ParamModel):
    p = int(getattr(mdl, "P", 1))
    et = np.zeros(shape=(p, mdl.N + 1))
    # et[:, 0] = 0.4 * (np.random.random(size=mdl.P) - 0.5) * mdl.eMax
    et[:, -1] = 0
    at: np.ndarray = np.asarray(pmt.mu / (mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)[:-1]), dtype=float)
    zt = np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(p, mdl.N - 1))
    for i, ai in enumerate(at):
        et[:, i + 1] = et[:, i] * (1.0 - ai * mdl.dt) + pmt.gamma * zt[:, i]
    return et


def _load_cn_data(csv_path: str, use_etf: bool) -> pd.DataFrame:
    feeder = cn_feeder
    if feeder is None:
        raise ImportError("dl.feeder.csv_feeder is not available")
    assert callable(feeder)
    return feeder(csv_path=csv_path, use_etf=use_etf)


def get_inputs(df: pd.DataFrame, length: int = 240) -> np.ndarray:
    """"""
    inputs = []
    for td, da in df.groupby("trading_date"):
        inp = da[["tau", "basis"]].to_numpy()
        if len(inp) == length:
            inputs.append(inp)
    return np.array(inputs)


def experiment1(engine: Any, num_epochs: int = 3000) -> List:
    print('  Iter       Loss        IB            CB        |   Time    LR')

    # Init timer and history list
    t0 = time.time()
    history = []
    for i in range(num_epochs):

        inputs = sim1(engine.mdl, engine.pmt)
        loss = engine.train_sim(inputs)
        ib, cb = engine.calc_boundaries(t=0.0)

        ct = time.time() - t0

        optim = getattr(engine, "optim", getattr(engine, "optim1", None))
        lr = optim.param_groups[0]["lr"] if optim is not None else 0.0
        hentry = (i, loss, ib, cb, ct, lr)
        history.append(hentry)

        if i % 100 == 0:
            print('{:5d} {:12.4f}  {:12.4f}  {:12.4f}   | {:6.1f} {:12.6f}'.format(*hentry))
    return history


def experiment2(engine: Any, num_epochs: int = 3000) -> List:
    # Initialize header
    print('  Iter       Loss        IL            IS            CL            CS        |   Time    LR')

    # Init timer and history list
    t0 = time.time()
    history = []
    for i in range(num_epochs):

        inputs = sim1(engine.mdl, engine.pmt)
        loss = engine.train_sim(inputs)
        eil, eis, ecl, ecs = engine.calc_boundaries(t=0.1)

        ct = time.time() - t0

        optim = getattr(engine, "optim1", getattr(engine, "optim", None))
        lr = optim.param_groups[0]["lr"] if optim is not None else 0.0
        hentry = (i, loss, eil, eis, ecl, ecs, ct, lr)
        history.append(hentry)

        if i % 100 == 0:
            print('{:5d} {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}   | {:6.1f} {:12.6f}'.format(*hentry))
    return history


def calc_reward(engine: Any, mdl: _GridModel, init_pos: float = 0.0) -> Tuple[float, float]:
    """"""
    tau = engine.mdl.T - np.linspace(0.0, engine.mdl.T, engine.mdl.N + 1)
    # inputs = sim3(engine.mkt, mdl, engine.pmt)
    inputs = sim6(engine.mkt, mdl, engine.pmt)
    # pre_pos = init_pos * np.ones(mdl.P)
    # cur_pos = init_pos * np.ones(mdl.P)
    pre_pos = np.zeros(mdl.P)
    cur_pos = np.zeros(mdl.P)
    rtn = np.zeros(mdl.P)
    dr = np.exp(engine.mkt.rf * engine.mdl.dt)
    for i in range(engine.mdl.N):
        open_pos = np.zeros(mdl.P)
        close_pos = np.zeros(mdl.P)
        et = inputs[:, i]
        de = inputs[:, i+1] - inputs[:, i]
        eil, eis, ecl, ecs = engine.calc_boundaries(t=tau[i])
        if init_pos >= 0:
            ils = np.where((pre_pos == 0) & (et >= eil))[0]
            cls = np.where((pre_pos == -1) & (et <= ecl))[0]
            cur_pos[ils] = -1
            cur_pos[cls] = 0
        if init_pos <= 0:
            iss = np.where((pre_pos == 0) & (et <= eis))[0]
            css = np.where((pre_pos == 1) & (et >= ecs))[0]
            cur_pos[iss] = 1
            cur_pos[css] = 0
        ops = np.where((cur_pos != pre_pos) & (cur_pos != 0))[0]
        cps = np.where((cur_pos != pre_pos) & (cur_pos == 0))[0]
        open_pos[ops] = 1
        close_pos[cps] = 1
        rtn = dr * (rtn + cur_pos * de - open_pos * engine.mkt.C0 - close_pos * engine.mkt.C1)
        pre_pos = cur_pos
    rtn = rtn - abs(cur_pos) * engine.mkt.C1
    rtn = rtn * np.exp(-engine.mkt.rf * engine.mdl.T)
    return np.mean(rtn), np.std(rtn)


def experiment_real(
        engine: Engine, csv_path: str, length: int = 240,
        start_date: int = 2010, end_date: int = 2020, use_etf: bool = False
):
    """"""
    # data = csv_feeder(csv_path=csv_path)
    # data = bn_feeder(csv_path=csv_path)
    if cn_feeder is None:
        raise ImportError("dl.feeder.csv_feeder is not available")
    data = _load_cn_data(csv_path=csv_path, use_etf=use_etf)

    # Initialize header
    print('  Iter       Loss        IL            IS            CL            CS        |   Time    LR')

    # Init timer and history list
    t0 = time.time()
    history = []
    count = 1
    for contract, df in data.groupby("contract"):
        if contract >= f"IF{end_date}" or contract <= f"IF{start_date}":
            continue
        # if "IF1501" <= contract <= "IF1701":
        #     continue
        inputs = get_inputs(df=df, length=length)
        loss = engine.train_real(inputs)
        eil, eis, ecl, ecs = engine.calc_boundaries(t=0.5)

        ct = time.time() - t0

        optim = getattr(engine, "optim1", getattr(engine, "optim", None))
        lr = optim.param_groups[0]["lr"] if optim is not None else 0.0
        hentry = (count, loss, eil, eis, ecl, ecs, ct, lr)
        history.append(hentry)

        print('{:5d} {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}   | {:6.1f} {:12.6f}'.format(*hentry))
        count += 1
    return history


def get_engine2(mkt: MarketData, mdl: ModelData, pmt: ParameterData, dv1_file: str, dv2_file: str):
    """"""
    engine2 = Engine2(mkt, mdl, pmt)
    engine2.dv1 = torch.load(dv1_file)
    engine2.dv2 = torch.load(dv2_file)
    return engine2


def get_boundaries(engine: Engine) -> pd.DataFrame:
    """"""
    df = pd.DataFrame(columns=["ILR", "ISR", "CLR", "CSR"])
    for i in range(engine.mdl.N):
        t = i * engine.mdl.dt
        ilr, isr, clr, csr = engine.calc_boundaries(t)
        df.loc[t] = [ilr, isr, clr, csr]
    return df


def res_plot(mod: Mod, bdy: BoundaryData):
    """"""
    if mod == Mod.LF:
        df = pd.read_csv("res_lf.csv", index_col=0)
    elif mod == Mod.SF:
        df = pd.read_csv("res_sf.csv", index_col=0)
    else:
        df = pd.read_csv("res_lsf.csv", index_col=0)

    df = df.loc[df.index <= 1000]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    xs = df.index

    ax[0].plot(xs, df["Loss"], 'k-')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title(r'Training Loss ($\\lambda = 10^{-3}$)')

    if mod in (Mod.LF, Mod.LSF):
        ax[1].plot(xs, [bdy.ILR for _ in range(len(xs))], 'k--')
        ax[1].plot(xs, [bdy.CLR for _ in range(len(xs))], 'k--')
        ax[1].plot(xs, df["IL"], 'b-', label="ILR")
        ax[1].plot(xs, df["CL"], 'g-', label="CLR")

    if mod in (Mod.SF, Mod.LSF):
        ax[1].plot(xs, [bdy.ISR for _ in range(len(xs))], 'k--')
        ax[1].plot(xs, [bdy.CSR for _ in range(len(xs))], 'k--')
        ax[1].plot(xs, df["IS"], 'b-.', label="ISR")
        ax[1].plot(xs, df["CS"], 'g-.', label="CSR")
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('$\\varepsilon$')
    ax[1].legend()
    ax[1].set_title(r'Boundaries ($\\lambda = 10^{-3}$)')

    fig.show()
    fig.savefig("rl_res.png")


def real_plot(csv_file: str):
    """"""
    df = pd.read_csv(csv_file, index_col=0)
    xs = df.index
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, np.zeros(len(xs)), 'k--')
    ax.plot(xs, df["IL"], 'b-', label="ILR")
    ax.plot(xs, df["CL"], 'r-', label="CLR")
    ax.plot(xs, df["IS"], 'b-.', label="ISR")
    ax.plot(xs, df["CS"], 'r-.', label="CSR")
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    ax.legend()
    ax.set_title(r'IF Contract Boundaries ($\\lambda = 10^{-3}$)')
    fig.show()


def boundary_plot(csv_file: str):
    """"""
    df = pd.read_csv(csv_file, index_col=0)
    # xs = df.index[-1] - df.index
    xs = df.index
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, np.zeros(len(xs)), 'k--')
    ax.plot(xs, df["ILR"], 'b-', label="ILR")
    ax.plot(xs, df["CLR"], 'r-', label="CLR")
    ax.plot(xs, df["ISR"], 'b-.', label="ISR")
    ax.plot(xs, df["CSR"], 'r-.', label="CSR")
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('$\\varepsilon$')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(r'IF Contract Boundaries ($\\lambda = 10^{-3}$)')
    fig.show()
    fig.savefig("IF_Contract_Boundaries.png")

