from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from surmount.data import FinancialStatement, EarningsSurprises, AnalystEstimates, LeveredDCF
import numpy as np
from datetime import datetime, timedelta

class TradingStrategy(Strategy):
    """
    Advanced fundamental analysis strategy that:
    1. Computes En and EAn metrics based on earnings, variance, and EBITDA
    2. Dynamically adds universe assets that perform in top 90th percentile for 3 consecutive periods
    3. Optimizes allocation based on fundamental metrics and DCF risk-adjusted returns
    4. Implements real-time price-based risk management
    """

    def __init__(self, universe=None):
        """
        Initialize strategy with universe of potential assets
        
        Args:
            universe: List of ticker symbols to consider for portfolio entry
        """
        # Initial portfolio assets
        self.portfolio_assets = ["SPY", "QQQ", "IWM"]
        
        # Universe of assets to evaluate for potential inclusion
        self.universe = universe if universe else ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
                                                     "META", "TSLA", "JPM", "V", "WMT"]
        
        # Track consecutive periods assets are in top 90th percentile
        self.universe_performance_tracker = {asset: 0 for asset in self.universe}
        
        # Track entry prices for position management
        self.entry_prices = {}
        
        # Track last rebalance date
        self.last_rebalance_date = None
        
        # Weights for metrics
        self.W1 = 0.5  # Weight for earnings surprise component
        self.W2 = 0.3  # Weight for earnings variance component
        self.W3 = 0.2  # Weight for EBITDA component
        
        # Allocation optimization weights
        self.B1_weight = 0.4  # Weight for En
        self.B2_weight = 0.6  # Weight for EAn

    @property
    def assets(self):
        """
        Returns combined list of portfolio and universe assets for data retrieval
        """
        return list(set(self.portfolio_assets + self.universe))

    @property
    def interval(self):
        """
        Use 1-minute interval for intraday price monitoring
        """
        return "1min"

    @property
    def data(self):
        """
        Specify all required data sources for the strategy
        """
        data_sources = []
        
        # Financial data for all assets (portfolio + universe)
        for ticker in self.assets:
            data_sources.extend([
                FinancialStatement(ticker),
                EarningsSurprises(ticker),
                AnalystEstimates(ticker),
                LeveredDCF(ticker)
            ])
        
        return data_sources

    def run(self, data):
        """
        Execute the trading strategy logic
        
        Args:
            data: Dictionary containing OHLCV and additional data sources
            
        Returns:
            TargetAllocation: Optimized portfolio allocations
        """
        try:
            ohlcv_data = data.get("ohlcv", [])
            
            if not ohlcv_data or len(ohlcv_data) < 2:
                log("Insufficient OHLCV data available")
                return TargetAllocation({})
            
            current_date = datetime.now()
            
            # Check if monthly rebalancing is needed (every 30 days)
            should_rebalance = self._should_rebalance_monthly(current_date)
            
            # Always check for intraday risk management triggers
            risk_adjustments = self._check_intraday_risk_triggers(ohlcv_data)
            
            if should_rebalance:
                log("Monthly rebalancing triggered")
                return self._monthly_rebalance(data, ohlcv_data)
            elif risk_adjustments:
                log("Intraday risk management triggered")
                return risk_adjustments
            else:
                # Maintain current allocations
                return self._get_current_allocation()
                
        except Exception as e:
            log(f"Error in strategy execution: {str(e)}")
            return TargetAllocation({})

    def _should_rebalance_monthly(self, current_date):
        """Check if 30 days have passed since last rebalance"""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= 30

    def _check_intraday_risk_triggers(self, ohlcv_data):
        """
        Check for intraday price change triggers and adjust positions accordingly
        
        Price change rules:
        - <= -10%: Exit position (100% sell)
        - >= 35%: Exit position (100% sell)
        - >= 25% and < 35%: Sell 35%
        - >= 15% and < 25%: Sell 25%
        - >= 10% and < 15%: Sell 15%
        - >= 30%: Sell 35%
        """
        if len(ohlcv_data) < 2:
            return None
        
        adjustments_needed = False
        new_allocations = {}
        
        for ticker in self.portfolio_assets:
            if ticker not in ohlcv_data[-1]:
                continue
            
            current_price = ohlcv_data[-1][ticker].get('close')
            entry_price = self.entry_prices.get(ticker)
            
            if not current_price or not entry_price:
                # Initialize entry price if not set
                self.entry_prices[ticker] = current_price
                continue
            
            # Calculate percentage change
            price_change = ((current_price - entry_price) / entry_price) * 100
            
            # Determine action based on price change
            if price_change <= -10 or price_change >= 35:
                # Exit position completely
                new_allocations[ticker] = 0
                adjustments_needed = True
                log(f"{ticker}: Price change {price_change:.2f}%, exiting position")
                
            elif price_change >= 25:
                # Sell 35%
                current_alloc = self._get_current_allocation().get(ticker, 0)
                new_allocations[ticker] = current_alloc * 0.65
                adjustments_needed = True
                log(f"{ticker}: Price change {price_change:.2f}%, selling 35%")
                
            elif price_change >= 15:
                # Sell 25%
                current_alloc = self._get_current_allocation().get(ticker, 0)
                new_allocations[ticker] = current_alloc * 0.75
                adjustments_needed = True
                log(f"{ticker}: Price change {price_change:.2f}%, selling 25%")
                
            elif price_change >= 10:
                # Sell 15%
                current_alloc = self._get_current_allocation().get(ticker, 0)
                new_allocations[ticker] = current_alloc * 0.85
                adjustments_needed = True
                log(f"{ticker}: Price change {price_change:.2f}%, selling 15%")
        
        if adjustments_needed:
            # Rebalance remaining liquidity across other assets
            return self._rebalance_after_sales(new_allocations)
        
        return None

    def _monthly_rebalance(self, data, ohlcv_data):
        """
        Perform monthly portfolio rebalancing with universe evaluation
        """
        self.last_rebalance_date = datetime.now()
        
        # Step 1: Compute En and EAn for all assets (portfolio + universe)
        asset_metrics = {}
        
        for ticker in self.assets:
            metrics = self._compute_asset_metrics(ticker, data)
            if metrics:
                asset_metrics[ticker] = metrics
        
        if not asset_metrics:
            log("Unable to compute metrics for any assets")
            return TargetAllocation({})
        
        # Step 2: Evaluate universe assets for portfolio inclusion
        self._evaluate_universe_inclusion(asset_metrics)
        
        # Step 3: Optimize allocation for portfolio assets
        allocations = self._optimize_allocation(asset_metrics, ohlcv_data, data)
        
        # Update entry prices for new allocations
        for ticker in allocations:
            if ticker in ohlcv_data[-1]:
                self.entry_prices[ticker] = ohlcv_data[-1][ticker].get('close')
        
        return TargetAllocation(allocations)

    def _compute_asset_metrics(self, ticker, data):
        """
        Compute En and EAn metrics for a given asset
        
        Returns:
            dict: {'En': float, 'EAn': float, 'RD': float}
        """
        try:
            # Get financial data
            financial_stmt = data.get(("financial_statement", ticker), [])
            earnings_surprises = data.get(("earnings_surprises", ticker), [])
            analyst_estimates = data.get(("analyst_estimates", ticker), [])
            dcf_data = data.get(("levered_dcf", ticker), [])
            
            if not financial_stmt or not earnings_surprises:
                return None
            
            # Extract latest earnings data
            latest_financial = financial_stmt[-1] if financial_stmt else {}
            latest_surprise = earnings_surprises[-1] if earnings_surprises else {}
            latest_estimates = analyst_estimates[-1] if analyst_estimates else {}
            
            eps_actual = latest_surprise.get('epsActual', latest_financial.get('eps'))
            eps_estimated = latest_surprise.get('epsEstimated')
            eps_current = latest_financial.get('eps')
            ebitda_current = latest_financial.get('ebitda')
            ebitda_avg = latest_estimates.get('ebitdaAvg')
            
            # Validate required data
            if None in [eps_actual, eps_estimated, eps_current]:
                return None
            
            # Calculate B1 (earnings surprise adjusted)
            if eps_actual != 0:
                B1 = 1 - ((eps_estimated - eps_actual) / eps_actual)
            else:
                B1 = 1
            
            # Calculate A1
            A1 = eps_current - (1 - B1)
            
            # Calculate B2 and A2 (earnings variance over 5 years)
            eps_history = [fs.get('eps') for fs in financial_stmt[-20:] if fs.get('eps')]  # Approximate 5 years
            
            if len(eps_history) >= 8:  # Need sufficient history
                variance_past = np.var(eps_history[:-1]) if len(eps_history) > 1 else 1
                variance_current = np.var(eps_history)
                
                B2 = -1 / variance_past if variance_past != 0 else 0
                A2 = 1 / variance_current if variance_current != 0 else 0
            else:
                B2 = 0
                A2 = 0
            
            # Calculate B3 and A3 (EBITDA vs estimates)
            if ebitda_current and ebitda_avg:
                B3 = ebitda_avg - ebitda_current
                A3 = ebitda_current - ebitda_avg
            else:
                B3 = 0
                A3 = 0
            
            # Calculate En and EAn (weighted averages)
            En = self.W1 * B1 + self.W2 * B2 + self.W3 * B3
            EAn = self.W1 * A1 + self.W2 * A2 + self.W3 * A3
            
            # Calculate RD (risk-adjusted return from DCF)
            RD = self._calculate_rd(dcf_data, data.get("ohlcv", []), ticker)
            
            return {
                'En': En,
                'EAn': EAn,
                'RD': RD,
                'score': self.B1_weight * En + self.B2_weight * EAn
            }
            
        except Exception as e:
            log(f"Error computing metrics for {ticker}: {str(e)}")
            return None

    def _calculate_rd(self, dcf_data, ohlcv_data, ticker):
        """
        Calculate risk-adjusted return (RD)
        RD = (DCF Stock Price / current price - 1) / Drawdown
        """
        try:
            if not dcf_data or not ohlcv_data:
                return 0
            
            dcf_price = dcf_data[-1].get('stockPrice') if dcf_data else None
            current_price = ohlcv_data[-1].get(ticker, {}).get('close') if ohlcv_data else None
            
            if not dcf_price or not current_price or current_price == 0:
                return 0
            
            # Calculate return
            price_return = (dcf_price / current_price - 1)
            
            # Calculate maximum drawdown over available history
            prices = [candle.get(ticker, {}).get('close', 0) for candle in ohlcv_data[-252:]]  # Last year
            prices = [p for p in prices if p > 0]
            
            if len(prices) < 2:
                return 0
            
            max_drawdown = self._calculate_max_drawdown(prices)
            
            # Risk-adjusted return
            if max_drawdown != 0:
                RD = price_return / abs(max_drawdown)
            else:
                RD = price_return
            
            return RD
            
        except Exception as e:
            log(f"Error calculating RD: {str(e)}")
            return 0

    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from price series"""
        if not prices or len(prices) < 2:
            return 0
        
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (price - peak) / peak if peak != 0 else 0
            max_dd = min(max_dd, dd)
        
        return max_dd

    def _evaluate_universe_inclusion(self, asset_metrics):
        """
        Evaluate universe assets for portfolio inclusion
        Assets must be in top 90th percentile for 3 consecutive periods
        """
        # Calculate 90th percentile threshold
        scores = [metrics['score'] for metrics in asset_metrics.values()]
        
        if not scores:
            return
        
        percentile_90 = np.percentile(scores, 90)
        
        # Update tracker for universe assets
        for ticker in self.universe:
            if ticker in asset_metrics:
                score = asset_metrics[ticker]['score']
                
                if score >= percentile_90:
                    self.universe_performance_tracker[ticker] += 1
                    
                    # Add to portfolio if in top 90th for 3 consecutive periods
                    if self.universe_performance_tracker[ticker] >= 3:
                        if ticker not in self.portfolio_assets:
                            self.portfolio_assets.append(ticker)
                            log(f"Adding {ticker} to portfolio (3 consecutive top 90th percentile)")
                else:
                    # Reset counter if not in top 90th
                    self.universe_performance_tracker[ticker] = 0

    def _optimize_allocation(self, asset_metrics, ohlcv_data, data):
        """
        Optimize portfolio allocation to maximize f(B1*En + B2*EAn)
        subject to maximizing RD
        """
        allocations = {}
        
        # Filter for portfolio assets only
        portfolio_metrics = {
            ticker: metrics for ticker, metrics in asset_metrics.items()
            if ticker in self.portfolio_assets
        }
        
        if not portfolio_metrics:
            return allocations
        
        # Calculate scores and RD for each asset
        scores = []
        rds = []
        tickers = []
        
        for ticker, metrics in portfolio_metrics.items():
            scores.append(metrics['score'])
            rds.append(max(metrics['RD'], 0))  # Ensure non-negative
            tickers.append(ticker)
        
        # Normalize scores to [0, 1]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score != min_score else 1
        
        normalized_scores = [(s - min_score) / score_range for s in scores]
        
        # Combine score and RD (prioritize RD as constraint)
        combined_scores = [
            ns * (1 + rd) for ns, rd in zip(normalized_scores, rds)
        ]
        
        # Allocate based on combined scores
        total_combined = sum(combined_scores) if sum(combined_scores) > 0 else 1
        
        for ticker, combined_score in zip(tickers, combined_scores):
            allocations[ticker] = combined_score / total_combined
        
        # Ensure allocations sum to 1
        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            allocations = {k: v / total_alloc for k, v in allocations.items()}
        
        return allocations

    def _rebalance_after_sales(self, partial_allocations):
        """
        Rebalance liquidity after partial sales across remaining assets
        """
        # Get assets that weren't sold
        remaining_assets = [
            ticker for ticker in self.portfolio_assets 
            if ticker not in partial_allocations or partial_allocations[ticker] > 0
        ]
        
        if not remaining_assets:
            return TargetAllocation({})
        
        # Calculate freed up allocation
        total_freed = 1 - sum(partial_allocations.values())
        
        # Distribute freed allocation equally among remaining assets
        additional_per_asset = total_freed / len(remaining_assets)
        
        final_allocations = {}
        for ticker in remaining_assets:
            current = partial_allocations.get(ticker, 1 / len(self.portfolio_assets))
            final_allocations[ticker] = current + additional_per_asset
        
        # Normalize
        total = sum(final_allocations.values())
        if total > 0:
            final_allocations = {k: v / total for k, v in final_allocations.items()}
        
        return TargetAllocation(final_allocations)

    def _get_current_allocation(self):
        """Return equal weight allocation for current portfolio"""
        if not self.portfolio_assets:
            return TargetAllocation({})
        
        equal_weight = 1.0 / len(self.portfolio_assets)
        return TargetAllocation({
            ticker: equal_weight for ticker in self.portfolio_assets
        })