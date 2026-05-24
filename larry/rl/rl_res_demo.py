import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from common import MarketData, RLModelData, ParameterData
from engines.rl_engine import RLEngine
from utils.experiments import experiment
from utils.util import get_boundaries


def run_engine(mkt: MarketData, mdl: RLModelData, pmt: ParameterData):
    """"""
    engine = RLEngine(mkt, mdl, pmt)
    history, bounds = experiment(engine, num_epochs=2000)
    return engine, history, bounds


if __name__ == "__main__":

    market_data = MarketData(rf=0.07, T=1.0, C0=1.20, C1=0.50)
    model_data = RLModelData(T=1.0, L=0.0001, P=512, N=50, I=1000, lr=0.0005)
    parameter_data = ParameterData(mu=2.28, gamma=0.3)

    eng, history, boundaries = run_engine(market_data, model_data, parameter_data)

    # torch.save(eng.dv1, "sim_dv1.pt")
    # torch.save(eng.dv2, "sim_dv2.pt")

    # df = get_boundaries(engine=eng)

    # df.to_csv("res_rl_boundaries.csv")