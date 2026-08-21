import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import RLModelData
from src.engines.rl_engine import RLEngine
from src.backtest.trader_common import combine_future_spot, combine_future_spot_new, get_stats


def trade_sim_contracts(engine: RLEngine, mdl: RLModelData, inputs_list: List[np.ndarray], rolling_num: int) -> Tuple[float, float, np.array]:
    """"""
    tsr = []
    for p in tqdm(range(mdl.P)):
        rn = rolling_num
        rtn = 0
        rtn_list = []
        count = 0
        for n in range(len(inputs_list)):
            inputs = inputs_list[n][p, :]
            pre_pos, cur_pos = 0, 0
            for i in range(rn, engine.mdl.N, 1):
                et = inputs[i]
                df = np.exp(-engine.mkt.rf * count * engine.mdl.dt)
                eil, eis, ecl, ecs = engine.calc_boundaries(t=i * engine.mdl.dt)
                if (pre_pos == 0) & (et >= eil):
                    cur_pos = -1
                elif (pre_pos == -1) & (et <= ecl):
                    cur_pos = 0
                elif (pre_pos == 0) & (et <= eis):
                    cur_pos = 1
                elif (pre_pos == 1) & (et >= ecs):
                    cur_pos = 0
                open_pos = cur_pos * (abs(pre_pos) - 1)
                close_pos = pre_pos * (abs(cur_pos) - 1)
                rtn += df * ((open_pos - close_pos) * et - abs(open_pos) * engine.mkt.C0 - abs(close_pos) * engine.mkt.C1)
                # rtn += df * mdl.L * cur_pos
                pre_pos = cur_pos
                rtn_list.append(rtn)
                count += 1
                if n != len(inputs_list) - 1:
                    if i == mdl.N - 1:
                        rn = rolling_num
                    elif (i >= mdl.N - 1 - rolling_num) & (cur_pos == 0):
                        rn = i - mdl.N + rolling_num + 1
                        break
        tsr.append(np.array(rtn_list))
    return np.array(tsr).reshape(mdl.P, -1).mean(axis=0)


def trade_one_contract(csv_path: str, contract_file: str, engine: RLEngine, use_etf: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """Trade a single contract using RL boundaries."""
    contract = contract_file.split(".")[0]
    data = combine_future_spot(csv_path, contract, dominant=True, use_etf=use_etf)
    data["basis1"] = data["future"] * np.exp(-0.025 * data["tau"]) - data["spot"]
    data["basis2"] = data["future"] - data["spot"]

    position = 0
    pos_list = []
    for dt in data.index:
        tau = data.loc[dt, "tau"]
        e = data.loc[dt, "basis1"]
        eil, eis, ecl, ecs = engine.calc_boundaries(tau)

        if (position == 0) & (e >= eil):
            position = -1
        elif (position == 0) & (e <= eis):
            position = 1
        elif ((position == -1) & (e <= ecl)) or ((position == 1) & (e >= ecs)):
            position = 0

        pos_list.append(position)

    sig_pos = pd.Series(pos_list, index=data.index)

    prc = data[["future", "spot"]]
    pos = pd.concat([sig_pos, -sig_pos], axis=1)
    pos.columns = ["future", "spot"]

    pnl = (prc.pct_change() * pos.shift(periods=1) - 0.0003 * abs(pos.diff())).mean(axis="columns").rename(
        f"{contract}_profit"
    )
    pos.rename(columns={"future": contract, "spot": "CSI300"}, inplace=True)
    return pnl, pos


def trade_one_contract_new(csv_path: str, contract_file: str, engine: RLEngine, use_etf: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """Trade a single contract using RL boundaries with ETF legs."""
    contract = contract_file.split(".")[0]
    data = combine_future_spot_new(csv_path, contract, dominant=True, use_etf=use_etf)
    data["basis1"] = data["future"] * np.exp(-0.025 * data["tau"]) - data["spot"]
    data["basis2"] = data["future"] - data["spot"]

    position = 0
    pos_list = []
    for dt in data.index:
        tau = data.loc[dt, "tau"]
        e = data.loc[dt, "basis1"]
        eil, eis, ecl, ecs = engine.calc_boundaries(tau)
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

    pnl = (prc.pct_change() * pos.shift(periods=1) - 0.000 * abs(pos.diff())).mean(axis="columns").rename(
        f"{contract}_profit"
    )
    pos.rename(columns={"future": contract, "spot": "CSI300"}, inplace=True)
    return pnl, pos


def trade_contracts(csv_path: str, start_year: int, engine: RLEngine, use_etf: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trade a set of contracts using RL boundaries."""
    fns = os.listdir(f"{csv_path}/IF/")
    df = pd.DataFrame()
    position = pd.DataFrame()
    stats = pd.DataFrame(columns=["mu", "vol", "pnl", "sr"])

    for fn in tqdm(sorted(fns)):
        contract = fn.split(".")[0]
        if int(contract[2:4]) >= start_year:
            data, pos = trade_one_contract(csv_path, fn, engine, use_etf)
            stats.loc[contract] = get_stats(data)
            df = pd.concat([df, data], axis=1)
            position = pd.concat([position, pos], axis=1)
            print(f"Complete {fn} File")

    stats.loc["IF Portfolio"] = get_stats(df.mean(axis=1))
    position.groupby(position.columns, axis=1).sum().to_csv("position.csv")
    return df.sort_index(), stats.sort_index()

