import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from common import MarketData, ParameterData, RLModelData
from engines.rl_engine import RLEngine


def get_boundaries(engine: RLEngine) -> pd.DataFrame:
    """"""
    df = pd.DataFrame(columns=["ILR", "ISR", "CLR", "CSR"])
    for i in range(engine.mdl.N):
        t = i * engine.mdl.dt
        ilr, isr, clr, csr = engine.calc_boundaries(t)
        df.loc[t] = [ilr, isr, clr, csr]
    return df


def get_real_inputs(df: pd.DataFrame, length: int = 240) -> np.array:
    """"""
    inputs = []
    for td, da in df.groupby("trading_date"):
        inp = da[["tau", "basis"]].to_numpy()
        if len(inp) == length:
            inputs.append(inp)
    return np.array(inputs)


def configure_plot_style() -> None:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Computer Modern Roman']
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['text.usetex'] = False