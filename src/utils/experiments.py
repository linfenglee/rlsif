import time
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

from engines.rl_engine import RLEngine
from utils.simulator import sim
from utils.util import get_boundaries


def experiment(engine: RLEngine, num_epochs: int = 3000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize header
    print('  Iter       Loss        IL            IS            CL            CS        |   Time    LR')

    # Init timer and history list
    t0 = time.time()
    history = []
    for i in range(num_epochs):

        inputs = sim(engine.mdl, engine.pmt)
        loss = engine.train_sim(inputs)
        eil, eis, ecl, ecs = engine.calc_boundaries(t=0.1)

        ct = time.time() - t0

        hentry = (i, loss, eil, eis, ecl, ecs, ct, engine.optim1.param_groups[0]["lr"])
        history.append(hentry)

        if i % 100 == 0:
            print('{:5d} {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}   | {:6.1f} {:12.6f}'.format(*hentry))

    history = pd.DataFrame(history, columns=["Iter", "Loss", "IL", "IS", "CL", "CS", "Time", "LR"]).set_index("Iter")
    bounds = get_boundaries(engine)

    return history, bounds


def experiment_repeats(engine_factory: Callable[[], RLEngine], num_epochs: int = 3000, num_repeats: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full experiments multiple times and aggregate loss mean/std by iteration.

    Each repeat creates a fresh engine via engine_factory to ensure independent training.
    """
    if num_repeats < 1:
        raise ValueError("num_repeats must be >= 1")

    histories: List[pd.DataFrame] = []
    bounds = None

    for _ in range(num_repeats):
        engine = engine_factory()
        history, bounds = experiment(engine, num_epochs=num_epochs)
        histories.append(history)

    loss_stack = np.stack([h["Loss"].to_numpy() for h in histories], axis=0)
    loss_mean = loss_stack.mean(axis=0)
    loss_std = loss_stack.std(axis=0) / np.sqrt(num_repeats)

    # Use means for other columns to keep a single summary dataframe
    base = histories[0].copy()
    base["LossMean"] = loss_mean
    base["LossStd"] = loss_std

    for col in ["IL", "IS", "CL", "CS", "Time", "LR"]:
        base[col] = np.mean([h[col].to_numpy() for h in histories], axis=0)

    base = base.drop(columns=["Loss"]).rename_axis("Iter")
    base = base[["LossMean", "LossStd", "IL", "IS", "CL", "CS", "Time", "LR"]]

    return base, bounds
