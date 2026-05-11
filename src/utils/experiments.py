import time
from typing import List

from engines.rl_engine import RLEngine
from utils.simulator import sim


def experiment(engine: RLEngine, num_epochs: int = 3000) -> List:
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
    return history