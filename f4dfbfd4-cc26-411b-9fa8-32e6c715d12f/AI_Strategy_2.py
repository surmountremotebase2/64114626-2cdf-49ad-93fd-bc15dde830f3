from surmount.base_class import Strategy, TargetAllocation
from surmount.data import Asset, EarningsSurprises, FinancialStatements, Estimates, DCF
from surmount.technical_indicators import ATR, SMA, Volume
import pandas as pd
import numpy as np

class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = self.get_sp500_tickers()  # Placeholder for actual S&P 500 tickers retrieval method
        self.earnings_surprises = [EarningsSurprises(ticker) for ticker in self.tickers]
        self.financial_statements = [FinancialStatements(ticker) for ticker in self.tickers]
        self.estimates = [Estimates(ticker) for ticker in self.tickers]
        self.dcf = [DCF(ticker) for ticker in self.tickers]
        self.data_list = self.earnings_surprises + self.financial_statements + self.estimates + self.dcf
        self.rebalance_interval = 30
        self.last_rebalance_day = None

    @property
    def assets(self):
        return self.tickers
    
    @property
    def interval(self):
        return "1day"
    
    @property
    def data(self):
        return self.data_list
    
    def get_sp500_tickers(self):
        # In actual implementation, fetch the S&P 500 tickers list from a reliable source
        return ["SPY", "AAPL", "MSFT", "..."]  # Placeholder tickers

    def run(self, data):
        # Check if it's time to rebalance
        today = pd.to_datetime('today')
        if self.last_rebalance_day is None or (today - self.last_rebalance_day).days >= self.rebalance_interval:
            self.last_rebalance_day = today
            # Filter on liquidity
            liquid_tickers = self.filter_on_liquidity(data)

            # Score and Rank
            ranked_tickers = self.rank_tickers(liquid_tickers, data)

            # Selection of top 10% for 3 consecutive periods - assume external tracking mechanism
            selected_tickers = self.select_top_tickers(ranked_tickers)
        else:
            # On non-rebalance days, enforce ATR-based stop-loss rules and take-profit exits for existing positions
            selected_tickers = self.enforce_exit_rules(data)
        
        # Compute allocations
        allocations = self.compute_allocations(selected_tickers, data)
        
        return TargetAllocation(allocations)

    def filter_on_liquidity(self, data):
        # Implement the 20-day average dollar volume liquidity filter
        liquid_tickers = []
        for ticker in self.tickers:
            try:
                avg_volume = SMA(ticker, data['ohlcv'], 20)
                avg_price = SMA(ticker, data['ohlcv'], 20, price_key='close')
                avg_dollar_volume = np.mean([v*p for v, p in zip(avg_volume, avg_price)])
                if avg_dollar_volume > SOME_DOLLAR_VOLUME_THRESHOLD:
                    liquid_tickers.append(ticker)
            except Exception as e:
                # Log or handle any exceptions such as missing data
                pass

        return liquid_tickers

    def rank_tickers(self, liquid_tickers, data):
        # Calculate composite scores (En + EAn) and rank tickers
        # Placeholder: dummy ranking mechanism
        ranked_tickers = sorted(liquid_tickers, key=lambda x: np.random.rand())
        return ranked_tickers
    
    def select_top_tickers(self, ranked_tickers):
        # Select top 10% tickers from ranked list
        top_tickers = ranked_tickers[:int(0.1 * len(ranked_tickers))]
        return top_tickers
    
    def enforce_exit_rules(self, data):
        # Apply ATR-based stop-loss and take-profit rules for existing positions
        # Placeholder: return existing positions unchanged; implement actual logic as per strategy requirement
        return self.current_positions

    def compute_allocations(self, selected_tickers, data):
        # Compute and normalize allocations based on fundamental scores, adjusting for partial profit-taking
        # Placeholder: evenly distribute allocations among selected tickers; implement actual scoring logic as required
        allocation_dict = {ticker: 1.0/len(selected_tickers) for ticker in selected_tickers}
        return allocation_dict

# This is a high-level approach to the strategy and requires filling in the specifics of each method based on the data sources and exact calculation methods for scoring, ranking, and exit rules.