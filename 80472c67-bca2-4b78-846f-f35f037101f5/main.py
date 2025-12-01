from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import ATR
from surmount.logging import log
from surmount.data import EarningsSurprises, FinancialStatement, FinancialEstimates, LeveredDCF
import numpy as np

class TradingStrategy(Strategy):
    """
    Improved Strategy with Liquidity Management and Monthly Rebalancing.
    
    Modifications for Risk/Slippage:
    1. Volume Filter: Only calculates scores for assets with > $10M daily dollar volume.
    2. Monthly Rebalance: Fundamental analysis and portfolio restructuring occur every 30 days.
    3. Daily Management: Stop losses and Take profits are monitored every single day.
    """

    def __init__(self):
        # --- UNIVERSE SETUP ---
        # Raw list (cleaned of duplicates and placeholders below)
        raw_tickers = [
            "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
            "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
            "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
            "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
            "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP",
            "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX", "BRK.B",
            "BBY", "TECH", "BIIB", "BLK", "BX", "BK", "BA", "BKNG", "BSX",
            "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "BXP", "CHRW", "CDNS",
            "CDW", "CE", "CF", "CHD", "CHTR", "CVX", "CMG",
            "CI", "CSCO", "CINF", "CTAS", "CME", "CVS", "CMA", "CLF", "CLX",
            "CMS", "CNA", "CNC", "CNSL", "COST", "COO", "COP", "CNX", "CNTY",
            "CSX", "CTIC", "CTSH", "CL", "CPB", "CERN", "CEG", "CFG",
            "CAH", "CTLT", "CHRD", "CBRE", "CNP", "K", "CRL", "CTVA", "HIG",
            "CPT", "CBT", "CIGI", "CBOE", "CMI", "AV", "CCL", "CHDW", "CPRT", "CARR", 
            "COF", "CP", "CAD", "CNI", "CZR", "KMX", "CAT", "COR", "SCHW", "C", 
            "KO", "COIN", "CMCSA", "CAG", "ED", "STZ", "GLW", "CPAY", "CSGP", "CTRA",
            "CRWD", "CCI", "DHR", "DRI", "DDOG", "DVA", "DAY", "DECK", "DE", "DELL", 
            "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR", "D", "DPZ", "DASH", 
            "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", 
            "EIX", "EW", "EA", "ELV", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", 
            "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", 
            "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", 
            "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "F", "FTNT", "FTV", "FOXA", 
            "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", 
            "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL", 
            "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT", "HOLX", "HD", "HON", 
            "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", 
            "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG", 
            "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", 
            "JNJ", "JCI", "JPM", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", 
            "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", 
            "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU", 
            "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", 
            "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", 
            "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", 
            "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", 
            "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", 
            "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", 
            "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH", "PAYX", 
            "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", 
            "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", 
            "PSA", "PHM", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", 
            "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", 
            "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", 
            "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", 
            "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", 
            "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN", "TPL", "TXT", 
            "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", 
            "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", 
            "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS", 
            "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS", 
            "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WSM", 
            "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"
        ]
        
        # Use set to remove duplicates, then sort for consistency
        self.tickers = sorted(list(set(raw_tickers)))

        # --- REBALANCE & RISK STATE ---
        self.rebalance_interval = 30  # Rebalance every 30 days
        self.days_since_rebalance = 30 # Start high to trigger initial allocation
        
        # Volume Filter Thresholds
        self.min_dollar_volume = 10_000_000 # $10M avg daily dollar volume minimum
        self.liquidity_lookback = 20 # Lookback for volume MA
        
        # --- ORIGINAL STRATEGY STATE ---
        self.holdings_info = {}
        self.percentile_streak = {}
        self.initial_prices = {}

        # Scoring weights
        self.W1 = 0.5
        self.W2 = 0.3
        self.W3 = 0.2
        self.Weight_En = 0.4
        self.Weight_EAn = 0.6

        # --- DATA LOADING ---
        self.data_list = []
        for ticker in self.tickers:
            self.data_list.append(EarningsSurprises(ticker))
            self.data_list.append(FinancialStatement(ticker))
            self.data_list.append(FinancialEstimates(ticker))
            self.data_list.append(LeveredDCF(ticker))

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def check_liquidity(self, ticker, ohlcv_data):
        """
        Calculates if the asset meets the liquidity threshold.
        Returns True if (Avg Volume * Current Price) > self.min_dollar_volume
        """
        if not ohlcv_data:
            return False
            
        # Get last N days of volume
        recent_bars = ohlcv_data[-self.liquidity_lookback:]
        if len(recent_bars) < 5: # Need at least some history
            return False
            
        volumes = [bar['volume'] for bar in recent_bars]
        avg_vol = np.mean(volumes)
        current_price = recent_bars[-1]['close']
        
        dollar_vol = avg_vol * current_price
        
        return dollar_vol >= self.min_dollar_volume

    def run(self, data):
        ohlcv = data.get("ohlcv", {})
        holdings = data.get("holdings", {})
        
        # If no price data, return empty
        if not ohlcv:
            return TargetAllocation({})

        # 1. --- DAILY RISK MANAGEMENT (Exits & Take Profits) ---
        # We must check this *every* day, not just on rebalance days.
        
        # Current Holdings in Surmount (assets with non-zero quantity)
        current_portfolio_tickers = set(k for k, v in holdings.items() if v > 0)
        # Track tickers tracked in our strategy logic
        tracked_tickers = set(self.holdings_info.keys())
        
        # Intersection: Tickers we think we hold AND actually hold
        active_holdings = current_portfolio_tickers.intersection(tracked_tickers)
        
        to_exit = set()
        partial_sells = {} # Map ticker -> new_allocation fraction relative to current

        for ticker in active_holdings:
            ticker_ohlcv = ohlcv.get(ticker, [])
            if not ticker_ohlcv: 
                continue
                
            last_bar = ticker_ohlcv[-1]
            current_price = last_bar['close']
            entry_price = self.holdings_info.get(ticker, {}).get('entry_price', current_price)
            
            # --- Stop Loss Logic (ATR Based) ---
            atr_series = ATR(ticker, ticker_ohlcv, length=14)
            atr_value = atr_series[-1] if atr_series else 0
            
            # Exit if price fell > 10% of ATR below entry (Tight Stop)
            if (current_price - entry_price) < (-0.10 * atr_value):
                to_exit.add(ticker)
                log(f"{ticker}: STOP LOSS triggered. Price: {current_price}, Entry: {entry_price}")
                continue

            # --- Fundamental Deterioration Exit ---
            # We assume deterioration check only happens on rebalance days or if we want to run it daily.
            # Given the original code ran it daily, we keep it daily but need scores.
            # To save compute, we might skip this unless it's a rebalance day, 
            # BUT original requirement says "keep initial logic". 
            # We will perform a lightweight check here or defer to rebalance. 
            # For efficiency, we will defer fundamental exits to the rebalance block
            # unless price action (Stop Loss) forces us out.

            # --- Profit Taking (Progressive Selling) ---
            pct_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Note: The original code logic for "selling fractions" implies reducing the *target allocation*.
            # Since we only generate a full TargetAllocation map at the end, we calculate the reduction factor here.
            sell_fraction = 0.0
            if pct_change >= 0.35: sell_fraction = 1.0
            elif pct_change >= 0.25: sell_fraction = 0.35
            elif pct_change >= 0.15: sell_fraction = 0.25
            elif pct_change >= 0.10: sell_fraction = 0.15
            
            if sell_fraction > 0:
                if sell_fraction == 1.0:
                    to_exit.add(ticker)
                    log(f"{ticker}: TAKE PROFIT - Full Exit (+35%)")
                else:
                    # Store the reduction factor to apply to existing allocation
                    partial_sells[ticker] = (1 - sell_fraction)
                    log(f"{ticker}: TAKE PROFIT - Selling {sell_fraction*100}% of position")

        # 2. --- REBALANCE TIMER & LIQUIDITY FILTER ---
        self.days_since_rebalance += 1
        is_rebalance_day = self.days_since_rebalance >= self.rebalance_interval

        # If it's NOT a rebalance day, we just want to maintain current positions
        # minus the exits/trims we calculated above.
        if not is_rebalance_day:
            # Get current allocation weights from the engine's perspective
            # NOTE: In Surmount, if we return TargetAllocation only for a subset, 
            # it might not work as intended depending on engine. safest is to reconstruct
            # targets based on current portfolio value.
            
            # Simplification: We return the same target allocation as yesterday 
            # but zero out the 'to_exit' and reduce 'partial_sells'.
            # However, since we don't store yesterday's exact target object, 
            # we can infer "Stay the Course" by returning None or empty? 
            # Surmount usually requires a TargetAllocation. 
            
            # We will reconstruct targets based on current holdings value
            total_portfolio_val = data.get("portfolio", {}).get("equity", 1.0) # avoid div by zero
            current_allocations = {}
            
            for ticker, val in holdings.items():
                if ticker in self.tickers: # Only manage our strategy assets
                    # Get current price to estimate current weight
                    # Assuming holdings[ticker] is quantity
                    try:
                        qty = val
                        if qty > 0:
                            price = ohlcv[ticker][-1]['close']
                            weight = (qty * price) / total_portfolio_val
                            current_allocations[ticker] = weight
                    except:
                        pass
            
            # Apply Exits
            final_targets = {}
            for t, w in current_allocations.items():
                if t in to_exit:
                    final_targets[t] = 0.0
                    if t in self.holdings_info: del self.holdings_info[t]
                elif t in partial_sells:
                    final_targets[t] = w * partial_sells[t]
                else:
                    final_targets[t] = w # Keep holding
            
            return TargetAllocation(final_targets)

        # 3. --- REBALANCING LOGIC (Only runs every 30 days) ---
        log("Performing Monthly Rebalance and Fundamental Scan...")
        self.days_since_rebalance = 0 # Reset timer

        universe_scores = {}
        
        # LIQUIDITY FIRST: Filter universe to only liquid assets to prevent slippage
        liquid_tickers = []
        for ticker in self.tickers:
            if self.check_liquidity(ticker, ohlcv.get(ticker, [])):
                liquid_tickers.append(ticker)
        
        log(f"Universe filtered by volume: {len(liquid_tickers)} of {len(self.tickers)} are liquid enough.")

        # Compute Scores for Liquid Tickers Only
        for ticker in liquid_tickers:
            scores = self.calculate_scores(ticker, data)
            if scores:
                combined = self.Weight_En * scores['En'] + self.Weight_EAn * scores['EAn']
                scores['combined'] = combined
                universe_scores[ticker] = scores
            else:
                universe_scores[ticker] = {'En': -999, 'EAn': -999, 'combined': -999}

        # Determine 90th percentile among liquid assets
        combined_list = [v['combined'] for v in universe_scores.values()]
        percentile_threshold = np.percentile(combined_list, 90) if combined_list else float('-inf')

        # Update Streak
        for ticker in liquid_tickers:
            scores = universe_scores.get(ticker)
            if scores['combined'] >= percentile_threshold:
                self.percentile_streak[ticker] = self.percentile_streak.get(ticker, 0) + 1
            else:
                self.percentile_streak[ticker] = 0

        # Eligibility: Top 10% for 3 periods
        eligible_entries = [t for t, count in self.percentile_streak.items() if count >= 3]
        
        # Candidate Assets = Current Holdings (that weren't stopped out) | Eligible New Entries
        # Note: We must exclude 'to_exit' generated in the Daily Risk check above
        current_holding_tickers = [t for t in self.holdings_info.keys() if t not in to_exit]
        candidate_assets = set(current_holding_tickers) | set(eligible_entries)

        # 4. --- FINAL ALLOCATION CALCULATION ---
        final_assets = []
        for ticker in candidate_assets:
            # Skip if we decided to exit today
            if ticker in to_exit: continue
            
            # Fundamental Exit Check (Rebalance Day Specific)
            # If metrics are negative and deteriorated, exit
            scores = universe_scores.get(ticker, {'En': 0, 'EAn': 0, 'combined': 0})
            
            # If we currently hold it, check if we should drop it fundamentally
            if ticker in self.holdings_info:
                current_price = ohlcv[ticker][-1]['close']
                func_val = self.func_DF(ticker, data, current_price)
                if func_val < percentile_threshold and max(scores['En'], scores['EAn']) < 0:
                    log(f"{ticker}: Exiting due to fundamental deterioration (Monthly Check)")
                    if ticker in self.holdings_info: del self.holdings_info[ticker]
                    continue

            final_assets.append(ticker)

        # Allocate based on scores
        alloc_scores = {}
        total_score = 0.0
        
        for ticker in final_assets:
            # If it's not in universe_scores (e.g. held asset that is no longer liquid or scored),
            # we give it a neutral/low score or force exit. 
            # Here we assume we keep it but don't add more weight if score missing.
            scores = universe_scores.get(ticker, {'combined': 0})
            score = max(0.0, scores['combined'])
            
            alloc_scores[ticker] = score
            total_score += score
            
            # Record entry if new
            if ticker not in self.holdings_info:
                self.holdings_info[ticker] = {'entry_price': ohlcv[ticker][-1]['close']}

        target_allocations = {}
        if total_score > 0:
            for ticker, score in alloc_scores.items():
                target_allocations[ticker] = score / total_score

        # Apply Partial Profit Taking limits to the NEW target allocations
        # If we are rebalancing into a stock we are also taking profit on, cap it.
        for ticker, reduction in partial_sells.items():
            if ticker in target_allocations:
                target_allocations[ticker] = target_allocations[ticker] * reduction

        # Final Normalization
        if target_allocations:
            total = sum(target_allocations.values())
            if total > 0:
                for t in target_allocations:
                    target_allocations[t] /= total

        return TargetAllocation(target_allocations)

    def calculate_scores(self, ticker, data):
        """
        Compute En and EAn. (Unchanged logic from original, just safer dict access)
        """
        try:
            earnings = data.get(("earnings_surprises", ticker))
            financials = data.get(("financial_statement", ticker))
            estimates = data.get(("financial_estimates", ticker))

            if not earnings or not financials or not estimates:
                return None

            def get_val(lst, key, index=-1):
                if not lst: return None
                if index < -len(lst) or index >= len(lst): return None
                return lst[index].get(key)

            eps_est = get_val(earnings, "epsEstimated")
            eps_act = get_val(earnings, "epsactual")
            B1 = (eps_est / eps_act) - 1.0 if (eps_est is not None and eps_act) else 0.0

            eps_act_n = get_val(financials, "eps")
            eps_est_prev = get_val(earnings, "epsEstimated", -2)
            A1 = eps_act_n - eps_est_prev if (eps_act_n is not None and eps_est_prev is not None) else 0.0

            eps_series = [d.get('eps') for d in financials[-13:] if d.get('eps') is not None]
            if len(eps_series) > 1:
                var_all = np.var(eps_series)
                var_hist = np.var(eps_series[:-1]) if len(eps_series) > 2 else var_all
                B2 = 1.0 / var_all if var_all != 0 else 0.0
                A2 = 1.0 / var_hist if var_hist != 0 else 0.0
            else:
                B2, A2 = 0.0, 0.0

            ebitda_est = get_val(estimates, "ebitdaAvg")
            ebitda_act = get_val(financials, "ebitda")
            B3 = (ebitda_est / ebitda_act) - 1.0 if (ebitda_est is not None and ebitda_act) else 0.0

            ebitda_est_prev = get_val(estimates, "ebitdaAvg", -2)
            A3 = ebitda_act - ebitda_est_prev if (ebitda_act is not None and ebitda_est_prev is not None) else 0.0

            En = (self.W1 * B1) + (self.W2 * B2) + (self.W3 * B3)
            EAn = (self.W1 * A1) + (self.W2 * A2) + (self.W3 * A3)

            return {'En': En, 'EAn': EAn}
        except Exception as exc:
            # log(f"Score error {ticker}: {exc}")
            return None

    def func_DF(self, ticker, data, current_price):
        """
        Compute custom func_DF metric. (Unchanged logic)
        """
        try:
            dcf_list = data.get(("levered_dcf", ticker))
            dcf_price = dcf_list[-1].get("Stock Price") if dcf_list else current_price

            inception_price = self.initial_prices.get(ticker)
            if inception_price is None:
                self.initial_prices[ticker] = current_price
                inception_price = current_price

            DD = dcf_price - inception_price
            numerator = (dcf_price / inception_price) - 1.0 if inception_price != 0 else 0.0

            if DD < 0:
                atr_series = ATR(ticker, data.get("ohlcv", {}).get(ticker, []), length=14)
                atr_val = atr_series[-1] if atr_series else 0.0
                denominator = DD * atr_val
                return numerator / denominator if denominator != 0 else numerator
            else:
                return numerator
        except Exception:
            return 0.0