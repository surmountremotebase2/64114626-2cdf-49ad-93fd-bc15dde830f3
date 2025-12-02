from surmount.base_class import Strategy, TargetAllocation

class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = ["AAPL","MSFT"]
        self.data_list = []

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def run(self, data):
        return TargetAllocation({"AAPL": 0.5, "MSFT": 0.5})