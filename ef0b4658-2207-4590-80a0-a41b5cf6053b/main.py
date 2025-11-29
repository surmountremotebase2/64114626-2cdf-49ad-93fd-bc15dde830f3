from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import SMA
from surmount.logging import log

class TradingStrategy(Strategy):
    @property
    def assets(self):
        return ["SPY"]

    @property
    def interval(self):
        return "1day"

    @property
    def data(self):
        return []

    def run(self, data):
        allocation = {}
        ticker = "SPY"
        
        # Se non ci sono dati, esci
        if "ohlcv" not in data or len(data["ohlcv"]) < 21:
            return TargetAllocation({})

        current_price = data["ohlcv"][-1][ticker]["close"]
        sma_list = SMA(ticker, data["ohlcv"], length=20)
        
        if not sma_list:
            return TargetAllocation({})
            
        current_sma = sma_list[-1]

        # Logica: Prezzo > Media Mobile = Compra (100%), altrimenti Vendi (0%)
        if current_price > current_sma:
            allocation[ticker] = 1.0
        else:
            allocation[ticker] = 0.0

        return TargetAllocation(allocation)