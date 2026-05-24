import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest.trader_common import combine_future_spot_new, get_stats


def trade_one_contract_fdm(csv_path: str, contract_file: str, boundaries: pd.DataFrame, use_etf: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """Trade a single contract using FDM boundaries."""
    contract = contract_file.split(".")[0]
    data = combine_future_spot_new(csv_path, contract, dominant=True, use_etf=use_etf)
    data["basis1"] = data["future"] * np.exp(-0.025 * data["tau"]) - data["spot"]

    position = 0
    pos_list = []
    all_indices = list(set(data["tau"]) | set(boundaries.index))
    bdr = boundaries.reindex(all_indices).sort_index()
    bdr = bdr.interpolate(method="linear")

    for dt in data.index:
        tau = data.loc[dt, "tau"]
        e = data.loc[dt, "basis1"]
        eil = bdr.loc[tau, "ILR"]
        eis = bdr.loc[tau, "ISR"]
        ecl = bdr.loc[tau, "CLR"]
        ecs = bdr.loc[tau, "CSR"]

        if (position == 0) & (e >= eil):
            position = -1
        elif (position == 0) & (e <= eis):
            position = 1
        elif ((position == -1) & (e <= ecl)) or ((position == 1) & (e >= ecs)):
            position = 0

        pos_list.append(position)

    sig_pos = pd.Series(pos_list, index=data.index)

    prc = data[["future", "etf"]]
    pos = pd.concat([sig_pos, -sig_pos], axis=1)
    pos.columns = ["future", "etf"]

    pnl = (prc.pct_change() * pos.shift(periods=1) - 0.0001 * abs(pos.diff())).mean(axis="columns").rename(
        f"{contract}_profit"
    )
    pos.rename(columns={"future": contract, "spot": "CSI300"}, inplace=True)
    return pnl, pos


def trade_contracts_fdm(csv_path: str, start_year: int, boundaries: pd.DataFrame, use_etf: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trade a set of contracts using FDM boundaries."""
    fns = os.listdir(f"{csv_path}/IF/")
    df = pd.DataFrame()
    position = pd.DataFrame()
    stats = pd.DataFrame(columns=["mu", "vol", "pnl", "sr"])

    for fn in tqdm(sorted(fns)):
        contract = fn.split(".")[0]
        if int(contract[2:4]) >= start_year:
            data, pos = trade_one_contract_fdm(csv_path, fn, boundaries, use_etf)
            stats.loc[contract] = get_stats(data)
            df = pd.concat([df, data], axis=1)
            position = pd.concat([position, pos], axis=1)
            print(f"Complete {fn} File")

    stats.loc["IF Portfolio"] = get_stats(df.mean(axis=1))
    position.groupby(position.columns, axis=1).sum().to_csv("position.csv")
    return df.sort_index(), stats.sort_index()

