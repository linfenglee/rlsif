from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd


def combine_future_spot(csv_path: str, contract: str, dominant: bool = False, use_etf: bool = False) -> pd.DataFrame:
    """Load futures/spot data and compute tau."""
    ir = pd.read_csv(f"{csv_path}yield_curve.csv", parse_dates=["trading_date"])[["trading_date", "1M"]]
    fut = pd.read_csv(f"{csv_path}IF/{contract}.csv", parse_dates=["datetime", "trading_date"])
    fut = fut.merge(ir, on="trading_date", how="inner").set_index("datetime")

    if use_etf:
        spot = pd.read_csv(f"{csv_path}510300_none.csv", parse_dates=["datetime"], index_col=1)
        df = pd.concat([fut[["close", "1M", "trading_date"]], 992 * spot["close"]], axis=1, join="inner")
    else:
        spot = pd.read_csv(f"{csv_path}000300.csv", parse_dates=["datetime"], index_col=1)
        df = pd.concat([fut[["close", "1M", "trading_date"]], spot["close"]], axis=1, join="inner")

    df.columns = ["future", "IR", "trading_date", "spot"]
    df["contract"] = contract
    df["tau"] = (df.index[-1] - df.index) / timedelta(days=365)

    if dominant:
        info = pd.read_csv(f"{csv_path}IF_INFO.csv", index_col=0, parse_dates=True)
        start = info.loc[contract, "start"]
        end = info.loc[contract, "end"]
        df = df.loc[start:end]

    return df.dropna()


def combine_future_spot_new(csv_path: str, contract: str, dominant: bool = False, use_etf: bool = False) -> pd.DataFrame:
    """Load futures/spot/ETF data and compute tau."""
    ir = pd.read_csv(f"{csv_path}yield_curve.csv", parse_dates=["trading_date"])[["trading_date", "1M"]]
    fut = pd.read_csv(f"{csv_path}IF/{contract}.csv", parse_dates=["datetime", "trading_date"])
    fut = fut.merge(ir, on="trading_date", how="inner").set_index("datetime")
    spot = pd.read_csv(f"{csv_path}000300.csv", parse_dates=["datetime"], index_col=1)
    df = pd.concat([fut[["close", "1M", "trading_date"]], spot["close"]], axis=1, join="inner")
    etf = pd.read_csv(f"{csv_path}510300_none.csv", parse_dates=["datetime"], index_col=1)
    df = pd.concat([df, etf["close"]], axis=1, join="inner")
    df.columns = ["future", "IR", "trading_date", "spot", "etf"]
    df["contract"] = contract
    df["tau"] = (df.index[-1] - df.index) / timedelta(days=365)

    if dominant:
        info = pd.read_csv(f"{csv_path}IF_INFO.csv", index_col=0, parse_dates=True)
        start = info.loc[contract, "start"]
        end = info.loc[contract, "end"]
        df = df.loc[start:end]

    return df.dropna()


def get_stats(series: pd.Series) -> List[float]:
    """Compute basic performance statistics for a PnL series."""
    mu = series.mean()
    vol = series.std()
    pnl = series.sum()
    sr = mu / (vol + 1e-6) * np.sqrt(4 * 60 * 365)
    return [mu, vol, pnl, sr]

