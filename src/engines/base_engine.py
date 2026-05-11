from abc import ABC


class BaseEngine(ABC):

    def __init__(self, market_data, model_data, parameter_data):
        """"""
        self.mkt = market_data
        self.mdl = model_data
        self.pmt = parameter_data