"""Backward-compatible re-exports for trader utilities.

Use src.backtest.trader_rl for RL-based trading and src.backtest.trader_fdm
for FDM-based trading. Common helpers live in src.backtest.trader_common.
"""

from src.backtest.trader_common import combine_future_spot, combine_future_spot_new, get_stats
from src.backtest.trader_fdm import trade_contracts_fdm, trade_one_contract_fdm
from src.backtest.trader_rl import trade_contracts, trade_one_contract, trade_one_contract_new

__all__ = [
    "combine_future_spot",
    "combine_future_spot_new",
    "get_stats",
    "trade_contracts",
    "trade_contracts_fdm",
    "trade_one_contract",
    "trade_one_contract_fdm",
    "trade_one_contract_new",
]







