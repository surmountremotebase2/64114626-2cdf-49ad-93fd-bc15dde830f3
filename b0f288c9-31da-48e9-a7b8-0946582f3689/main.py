from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log

class TradingStrategy(Strategy):
    @property
    def assets(self):
        # Testiamo con un solo asset per sbloccare l'interfaccia
        return ["SPY"]

    @property
    def interval(self):
        return "1day"

    @property
    def data(self):
        # Nessun dato fondamentale per ora, solo prezzo
        return []

    def run(self, data):
        # Compra e tieni SPY al 100%
        return TargetAllocation({"SPY": 1.0})