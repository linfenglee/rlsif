import numpy as np

from common import RLModelData, ParameterData


# def sim(mdl: RLModelData, pmt: ParameterData):
#     """"""
#     tau = mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)
#     zt = np.insert(
#         np.cumsum(np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(mdl.P, mdl.N)), axis=1),
#         0, 0, axis=1
#     )
#     et = pmt.gamma * np.power(tau, 1.0 - pmt.mu) * zt / (2 * pmt.mu - 1)
#     return et


def sim(mdl: RLModelData, pmt: ParameterData, scaler: float = 1.0):
    et = np.zeros(shape=(mdl.P, mdl.N + 1))
    et[:, 0] = np.random.normal(loc=0.0, scale=scaler * np.sqrt(mdl.T), size=mdl.P)
    at = pmt.mu / (mdl.T - np.linspace(0.0, mdl.T, mdl.N + 1)[:-1])
    zt = np.random.normal(loc=0.0, scale=np.sqrt(mdl.dt), size=(mdl.P, mdl.N - 1))
    for i in range(mdl.N - 1):
        et[:, i + 1] = et[:, i] * (1.0 - at[i] * mdl.dt) + pmt.gamma * zt[:, i]
    return et