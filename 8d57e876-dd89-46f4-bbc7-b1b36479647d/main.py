#Type code here
# Full strategy - cleaned + robust stop-loss + coherent fundamental rebalancing
# (ATR stop fixed, monthly fundamental exit logic fixed, stable non-rebalance targeting via last_targets)

from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import ATR
from surmount.logging import log
from surmount.data import EarningsSurprises, FinancialStatement, FinancialEstimates, LeveredDCF
import numpy as np


class TradingStrategy(Strategy):
    def __init__(self):
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

        # Remove duplicates + sort
        self.tickers = sorted(list(set(raw_tickers)))

        # Rebalance
        self.rebalance_interval = 30
        self.days_since_rebalance = 30  # trigger at start

        # Liquidity filter
        self.min_dollar_volume = 10_000_000
        self.liquidity_lookback = 20

        # State
        self.holdings_info = {}       # {ticker: {entry_price:..., peak_price:...}}
        self.percentile_streak = {}   # streak counter for top decile
        self.initial_prices = {}      # inception price for func_DF
        self.last_targets = {}        # stable targets between rebalances

        # Score weights
        self.W1 = 0.5
        self.W2 = 0.3
        self.W3 = 0.2
        self.Weight_En = 0.4
        self.Weight_EAn = 0.6

        # Data requests
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
        if not ohlcv_data:
            return False

        recent_bars = ohlcv_data[-self.liquidity_lookback:]
        if len(recent_bars) < 5:
            return False

        volumes = [bar.get("volume", 0) for bar in recent_bars if bar.get("volume") is not None]
        if not volumes:
            return False

        avg_vol = float(np.mean(volumes))
        current_price = recent_bars[-1].get("close", None)
        if current_price is None:
            return False

        dollar_vol = avg_vol * float(current_price)
        return dollar_vol >= self.min_dollar_volume

    def run(self, data):
        ohlcv = data.get("ohlcv", {})
        holdings = data.get("holdings", {})

        if not ohlcv:
            return TargetAllocation({})

        # -------------------------
        # 1) DAILY RISK MANAGEMENT
        # -------------------------
        current_portfolio_tickers = set(k for k, v in holdings.items() if v > 0)
        tracked_tickers = set(self.holdings_info.keys())
        active_holdings = current_portfolio_tickers.intersection(tracked_tickers)

        to_exit = set()
        partial_sells = {}  # ticker -> remaining multiplier (e.g., 0.75)

        for ticker in active_holdings:
            ticker_ohlcv = ohlcv.get(ticker, [])
            if not ticker_ohlcv:
                continue

            last_bar = ticker_ohlcv[-1]
            current_price = last_bar.get("close", None)
            if current_price is None:
                continue

            info = self.holdings_info.get(ticker, {})
            entry_price = info.get("entry_price", current_price)

            # --- STOP LOSS (ATR CORRECT) ---
            atr_series = ATR(ticker, ticker_ohlcv, length=14)
            atr_value = atr_series[-1] if atr_series else 0.0
            if atr_value and atr_value > 0:
                k_stop = 2.5
                use_trailing = True

                if use_trailing:
                    peak = info.get("peak_price", entry_price)
                    peak = max(peak, current_price)
                    self.holdings_info[ticker]["peak_price"] = peak
                    stop_level = peak - k_stop * atr_value
                else:
                    stop_level = entry_price - k_stop * atr_value

                if current_price < stop_level:
                    to_exit.add(ticker)
                    log(f"{ticker}: ATR STOP. Price={current_price:.2f}, Stop={stop_level:.2f}, ATR={atr_value:.2f}")
                    continue

            # --- PROFIT TAKING (PROGRESSIVE) ---
            pct_change = (current_price - entry_price) / entry_price if entry_price else 0.0

            sell_fraction = 0.0
            if pct_change >= 0.35:
                sell_fraction = 1.0
            elif pct_change >= 0.25:
                sell_fraction = 0.35
            elif pct_change >= 0.15:
                sell_fraction = 0.25
            elif pct_change >= 0.10:
                sell_fraction = 0.15

            if sell_fraction > 0:
                if sell_fraction >= 1.0:
                    to_exit.add(ticker)
                    log(f"{ticker}: TAKE PROFIT - Full Exit (+35% or more)")
                else:
                    partial_sells[ticker] = (1.0 - sell_fraction)
                    log(f"{ticker}: TAKE PROFIT - Selling {sell_fraction*100:.0f}% of position")

        # ---------------------------------
        # 2) REBALANCE TIMER
        # ---------------------------------
        self.days_since_rebalance += 1
        is_rebalance_day = self.days_since_rebalance >= self.rebalance_interval

        # Non-rebalance day: keep last targets, apply exits/trims
        if not is_rebalance_day:
            # Start from last known target allocations (stable)
            targets = dict(self.last_targets) if self.last_targets else {}

            # If no stored targets yet, fall back to "hold what you have" = do nothing
            if not targets:
                targets = {}

            # Apply exits + trims to targets
            for t in list(targets.keys()):
                if t in to_exit:
                    targets[t] = 0.0
                    if t in self.holdings_info:
                        del self.holdings_info[t]
                elif t in partial_sells:
                    targets[t] = targets[t] * partial_sells[t]

            # Normalize (optional, but good if trims happened)
            total = sum(w for w in targets.values() if w and w > 0)
            if total > 0:
                for t in targets:
                    if targets[t] > 0:
                        targets[t] /= total

            self.last_targets = dict(targets)
            return TargetAllocation(targets)

        # ---------------------------------
        # 3) MONTHLY REBALANCING + FUNDAMENTALS
        # ---------------------------------
        log("Performing Monthly Rebalance and Fundamental Scan...")
        self.days_since_rebalance = 0

        # Liquidity filter
        liquid_tickers = []
        for ticker in self.tickers:
            if self.check_liquidity(ticker, ohlcv.get(ticker, [])):
                liquid_tickers.append(ticker)

        log(f"Universe filtered by volume: {len(liquid_tickers)} of {len(self.tickers)} are liquid enough.")

        # Compute scores
        universe_scores = {}
        for ticker in liquid_tickers:
            scores = self.calculate_scores(ticker, data)
            if scores:
                combined = self.Weight_En * scores["En"] + self.Weight_EAn * scores["EAn"]
                scores["combined"] = combined
                universe_scores[ticker] = scores
            else:
                universe_scores[ticker] = {"En": -999, "EAn": -999, "combined": -999}

        combined_list = [v["combined"] for v in universe_scores.values() if v["combined"] > -900]
        percentile_threshold = np.percentile(combined_list, 90) if combined_list else float("-inf")

        # Update streak (top 10% for 3 consecutive rebalances)
        for ticker in liquid_tickers:
            scores = universe_scores.get(ticker, {"combined": -999})
            if scores["combined"] >= percentile_threshold:
                self.percentile_streak[ticker] = self.percentile_streak.get(ticker, 0) + 1
            else:
                self.percentile_streak[ticker] = 0

        eligible_entries = [t for t, count in self.percentile_streak.items() if count >= 3]

        # Candidates = current holdings (not stopped) + eligible new entries
        current_holding_tickers = [t for t in self.holdings_info.keys() if t not in to_exit]
        candidate_assets = set(current_holding_tickers) | set(eligible_entries)

        # -----------------------------
        # 4) FUNDAMENTAL REBALANCING (FIXED)
        # -----------------------------
        final_assets = []

        # Precompute list for percentile ranks (coherent, same scale: combined)
        combined_list_for_rank = combined_list[:]  # already filtered > -900

        def pct_rank_of_value(val, arr):
            if not arr:
                return 0.0
            # percentile rank: % of values <= val
            return (sum(1 for x in arr if x <= val) / len(arr)) * 100.0

        for ticker in candidate_assets:
            if ticker in to_exit:
                continue

            # If no OHLCV today, skip
            ticker_ohlcv = ohlcv.get(ticker, [])
            if not ticker_ohlcv or ticker_ohlcv[-1].get("close") is None:
                # If we held it, safer to drop on rebalance
                if ticker in self.holdings_info:
                    log(f"{ticker}: Exiting due to missing OHLCV/price on rebalance day.")
                    del self.holdings_info[ticker]
                continue

            # Score may be missing if not liquid / missing fundamentals
            scores = universe_scores.get(ticker, None)

            # If we currently hold it but it has missing/invalid fundamentals now -> exit for robustness
            if scores is None or scores.get("combined", -999) <= -900:
                if ticker in self.holdings_info:
                    log(f"{ticker}: Exiting due to missing/invalid fundamentals on rebalance day.")
                    del self.holdings_info[ticker]
                # If it's an eligible entry but missing score, also skip
                continue

            combined = scores["combined"]
            pct_rank = pct_rank_of_value(combined, combined_list_for_rank)

            current_price = float(ticker_ohlcv[-1]["close"])
            df_val = self.func_DF(ticker, data, current_price)

            # Deterioration rules (coherent)
            floor_pct = 60.0
            earnings_red = (scores.get("En", 0.0) < 0.0 and scores.get("EAn", 0.0) < 0.0)
            dcf_red = (df_val < 0.0)  # simple, stable

            if ticker in self.holdings_info:
                if (pct_rank < floor_pct and earnings_red) or dcf_red:
                    log(
                        f"{ticker}: Fundamental EXIT. pct={pct_rank:.1f}, "
                        f"En={scores['En']:.3f}, EAn={scores['EAn']:.3f}, DF={df_val:.3f}"
                    )
                    del self.holdings_info[ticker]
                    continue

            final_assets.append(ticker)

        # -----------------------------
        # 5) FINAL ALLOCATION (score-based)
        # -----------------------------
        alloc_scores = {}
        total_score = 0.0

        for ticker in final_assets:
            scores = universe_scores.get(ticker, {"combined": 0.0})
            score = float(scores.get("combined", 0.0))
            score = max(0.0, score)  # long-only

            alloc_scores[ticker] = score
            total_score += score

            # record entry for new positions
            if ticker not in self.holdings_info:
                entry = float(ohlcv[ticker][-1]["close"])
                self.holdings_info[ticker] = {"entry_price": entry, "peak_price": entry}

        target_allocations = {}
        if total_score > 0:
            for ticker, score in alloc_scores.items():
                target_allocations[ticker] = score / total_score

        # Apply partial profit-taking caps (if we are trimming a winner on the same rebalance)
        for ticker, reduction in partial_sells.items():
            if ticker in target_allocations:
                target_allocations[ticker] *= reduction

        # Normalize
        total = sum(target_allocations.values())
        if total > 0:
            for t in target_allocations:
                target_allocations[t] /= total

        # Store last targets for non-rebalance days
        self.last_targets = dict(target_allocations)
        return TargetAllocation(target_allocations)

    def calculate_scores(self, ticker, data):
        try:
            earnings = data.get(("earnings_surprises", ticker))
            financials = data.get(("financial_statement", ticker))
            estimates = data.get(("financial_estimates", ticker))

            if not earnings or not financials or not estimates:
                return None

            def get_val(lst, key, index=-1):
                if not lst:
                    return None
                if index < -len(lst) or index >= len(lst):
                    return None
                return lst[index].get(key)

            # Earnings surprises keys can vary; be defensive
            eps_est = get_val(earnings, "epsEstimated")
            eps_act = get_val(earnings, "epsactual")
            if eps_act is None:
                eps_act = get_val(earnings, "epsActual")

            B1 = (eps_est / eps_act) - 1.0 if (eps_est is not None and eps_act not in (None, 0)) else 0.0

            eps_act_n = get_val(financials, "eps")
            eps_est_prev = get_val(earnings, "epsEstimated", -2)
            A1 = (eps_act_n - eps_est_prev) if (eps_act_n is not None and eps_est_prev is not None) else 0.0

            eps_series = [d.get("eps") for d in financials[-13:] if d.get("eps") is not None]
            if len(eps_series) > 1:
                var_all = float(np.var(eps_series))
                var_hist = float(np.var(eps_series[:-1])) if len(eps_series) > 2 else var_all
                B2 = 1.0 / var_all if var_all != 0 else 0.0
                A2 = 1.0 / var_hist if var_hist != 0 else 0.0
            else:
                B2, A2 = 0.0, 0.0

            ebitda_est = get_val(estimates, "ebitdaAvg")
            ebitda_act = get_val(financials, "ebitda")
            B3 = (ebitda_est / ebitda_act) - 1.0 if (ebitda_est is not None and ebitda_act not in (None, 0)) else 0.0

            ebitda_est_prev = get_val(estimates, "ebitdaAvg", -2)
            A3 = (ebitda_act - ebitda_est_prev) if (ebitda_act is not None and ebitda_est_prev is not None) else 0.0

            En = (self.W1 * B1) + (self.W2 * B2) + (self.W3 * B3)
            EAn = (self.W1 * A1) + (self.W2 * A2) + (self.W3 * A3)

            return {"En": En, "EAn": EAn}
        except Exception:
            return None

    def func_DF(self, ticker, data, current_price):
        # Custom DCF-health metric; kept stable with safeguards
        try:
            dcf_list = data.get(("levered_dcf", ticker))
            dcf_price = dcf_list[-1].get("Stock Price") if dcf_list else current_price
            dcf_price = float(dcf_price) if dcf_price is not None else float(current_price)

            inception_price = self.initial_prices.get(ticker)
            if inception_price is None:
                self.initial_prices[ticker] = float(current_price)
                inception_price = float(current_price)

            inception_price = float(inception_price)
            if inception_price == 0:
                return 0.0

            DD = dcf_price - inception_price
            numerator = (dcf_price / inception_price) - 1.0

            if DD < 0:
                atr_series = ATR(ticker, data.get("ohlcv", {}).get(ticker, []), length=14)
                atr_val = atr_series[-1] if atr_series else 0.0
                atr_val = float(atr_val) if atr_val else 0.0

                # Avoid blow-ups
                denom = DD * max(atr_val, 1e-6)
                if abs(denom) < 1e-6:
                    return float(numerator)
                return float(numerator) / float(denom)
            else:
                return float(numerator)
        except Exception:
            return 0.0