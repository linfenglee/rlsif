import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def profit_plot(contract: str, df: pd.DataFrame):
    """"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df[f"{contract}_profit"].cumsum(), "b-")
    ax.set_xlabel('datetime')
    ax.set_ylabel('profit')
    ax.set_title(f'{contract} Contract Profit')
    fig.show()


def portfolio_profit_plot(df: pd.DataFrame):
    """"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Computer Modern Roman']
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['text.usetex'] = False

    fig, ax = plt.subplots(figsize=(15, 10))
    # ax.plot(df.index, df.mean(axis=1).cumsum(), "b-")
    # ax.plot(df.index, (1 + df.mean(axis=1)).cumprod(), "b-")
    (1 + df.mean(axis=1).dropna()).cumprod().plot(ax=ax)
    print(f"SR: {np.sqrt(250 * 60 * 4) * df.mean(axis=1).mean() / df.mean(axis=1).std()}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Wealth')
    # ax.set_title(f'Portfolio Wealth')
    fig.show()