import pandas as pd

from engines.rl_engine import RLEngine


def get_boundaries(engine: RLEngine) -> pd.DataFrame:
    """"""
    df = pd.DataFrame(columns=["ILR", "ISR", "CLR", "CSR"])
    for i in range(engine.mdl.N):
        t = i * engine.mdl.dt
        ilr, isr, clr, csr = engine.calc_boundaries(t)
        df.loc[t] = [ilr, isr, clr, csr]
    return df