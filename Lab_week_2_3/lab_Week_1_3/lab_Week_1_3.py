"""
Full Trading Lab (Week1–3) – 6 strategies:
1) SMA crossover (AAPL)
2) Bollinger mean-reversion (AAPL)
3) RSI active (multi-ticker)
4) MACD crossover (AAPL)
5) Growth vs Value fundamental
6) Momentum cross-sectional (multi-ticker)
Outputs: CSVs + individual equity charts + combined chart
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
import shutil

# =====================================================
# 1) DOWNLOAD MARKET DATA
# =====================================================
def download_ohlcv(tickers, start, end=None, interval='1d'):
    if end is None:
        end = dt.date.today().isoformat()
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False)

        if df.empty:
            print(f"WARNING: No data for {t}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        data[t] = df[["Open","High","Low","Close","Adj Close","Volume"]].copy()
    return data

# =====================================================
# 2) FUNDAMENTALS
# =====================================================
def fetch_basic_fundamentals(tickers):
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append({
            "ticker": t,
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "epsTTM": info.get("trailingEps"),
            "revenueTTM": info.get("totalRevenue"),
            "dividendYield": info.get("dividendYield"),
            "pbRatio": info.get("priceToBook"),
            "beta": info.get("beta"),
            "sector": info.get("sector")
        })
    return pd.DataFrame(rows).set_index("ticker")

# =====================================================
# 3) INDICATORS
# =====================================================
def SMA(x, win):
    return x.rolling(win, min_periods=1).mean()

def BollingerBands(x, win=20, n=2):
    ma = x.rolling(win, min_periods=1).mean()
    sd = x.rolling(win, min_periods=1).std()
    return pd.DataFrame({
        "bb_ma": ma,
        "bb_upper": ma + n*sd,
        "bb_lower": ma - n*sd
    })

def RSI(x, win=14):
    d = x.diff()
    up = d.where(d > 0, 0)
    dn = (-d).where(d < 0, 0)
    avg_up = up.ewm(alpha=1/win, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/win, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - 100/(1 + rs)

def MACD(x, fast=12, slow=26, signal=9):
    ema_fast = x.ewm(span=fast, adjust=False).mean()
    ema_slow = x.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "signal": signal_line, "hist": hist})

# =====================================================
# 4) STRATEGIES
# =====================================================
def sma_crossover_signals(price, short=50, long=200):
    s = SMA(price, short)
    l = SMA(price, long)
    sig = (s > l).astype(int)
    return sig.shift(1).fillna(0)

def bollinger_mean_reversion_signals(price, win=20, n=2):
    bb = BollingerBands(price, win, n)
    long_sig = (price < bb["bb_lower"]).astype(int)
    exit_sig = (price > bb["bb_ma"]).astype(int)

    pos = []
    holding = 0
    for i in range(len(price)):
        if long_sig.iloc[i] and holding == 0:
            holding = 1
        elif exit_sig.iloc[i] and holding == 1:
            holding = 0
        pos.append(holding)

    return pd.Series(pos, index=price.index).shift(1).fillna(0)

def classify_growth_value(fund):
    df = fund.copy()
    df["is_low_pe"] = (df["trailingPE"] < 20)
    df["growth_score"] = (df["epsTTM"] > 0).astype(int) + df["revenueTTM"].notna().astype(int)
    df["value_score"] = df["is_low_pe"].astype(int) + ((df["pbRatio"] < 3).fillna(False).astype(int))

    def tag(r):
        if r["growth_score"] >= 1 and r["value_score"] == 0:
            return "growth"
        if r["value_score"] >= 1 and r["growth_score"] == 0:
            return "value"
        return "neutral"

    df["tag"] = df.apply(tag, axis=1)
    return df

def momentum_cross_sectional(prices, lookback=6, top=3, bottom=3):
    monthly = prices.resample("ME").last()
    mom = monthly.pct_change(lookback)
    rebal_dates = mom.index[lookback:]

    weights, returns = [], []

    for d in rebal_dates:
        s = mom.loc[d].dropna().sort_values(ascending=False)
        long = s.index[:top]
        short = s.index[-bottom:]

        w = pd.Series(0, index=prices.columns)
        w.loc[long] = 1/len(long)
        w.loc[short] = -1/len(short)
        weights.append(w)

        idx = monthly.index.get_loc(d)
        if idx + 1 < len(monthly):
            nxt = monthly.index[idx + 1]
            r = (monthly.loc[nxt] / monthly.loc[d] - 1)
            returns.append((w * r).sum())
        else:
            returns.append(0)

    return pd.DataFrame(weights, index=rebal_dates), pd.Series(returns, index=rebal_dates)

def rsi_strategy(prices, win=14, low=30, high=70):
    rsi_df = prices.apply(RSI, win=win)
    pos = (rsi_df < low).astype(int) - (rsi_df > high).astype(int)
    returns = prices.pct_change().fillna(0)
    portfolio = (pos * returns).mean(axis=1)
    return (1 + portfolio).cumprod()

def macd_signals(price):
    macd_df = MACD(price)
    sig = (macd_df["macd"] > macd_df["signal"]).astype(int)
    return sig.shift(1).fillna(0)

# =====================================================
# BACKTEST
# =====================================================
def backtest_single(price, pos):
    ret = price.pct_change().fillna(0)
    pos = pos.ffill().fillna(0)
    equity = (1 + pos * ret).cumprod()
    equity[equity <= 0] = np.nan
    return equity

# =====================================================
# PERFORMANCE SUMMARY
# =====================================================
def performance_summary(eq):
    # Loại bỏ NaN
    if eq is None:
        return {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Max Drawdown (%)": 0.0
        }

    eq = eq.dropna()

    # Không đủ dữ liệu
    if eq.shape[0] < 2:
        return {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Max Drawdown (%)": 0.0
        }

    # Total return
    total = eq.iloc[-1] / eq.iloc[0] - 1

    # Years
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0:
        cagr = 0
    else:
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1

    # Max drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1).min()

    return {
        "Total Return (%)": total * 100,
        "CAGR (%)": cagr * 100,
        "Max Drawdown (%)": dd * 100
    }

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    tickers = ["AAPL","MSFT","AMZN","GOOGL","TSLA","META","JNJ","BAC","XOM","WMT"]
    start = "2015-01-01"

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs")

    print("Downloading price data...")
    data = download_ohlcv(tickers, start)
    prices = pd.DataFrame({t: data[t]["Adj Close"] for t in data})
    aapl = prices["AAPL"]

    print("Downloading fundamentals...")
    fund = fetch_basic_fundamentals(tickers)
    fv = classify_growth_value(fund)

    # STRATEGIES
    eq_sma = backtest_single(aapl, sma_crossover_signals(aapl))
    eq_bb = backtest_single(aapl, bollinger_mean_reversion_signals(aapl))

    ret = prices.pct_change().dropna()
    growth = fv[fv["tag"]=="growth"].index
    value = fv[fv["tag"]=="value"].index
    g_ret = ret[growth].mean(axis=1)
    v_ret = ret[value].mean(axis=1)
    eq_gv = (1 + (g_ret - v_ret)).cumprod()

    weights, port_rets = momentum_cross_sectional(prices)
    eq_mom = (1 + port_rets).cumprod().reindex(eq_sma.index, method="ffill")

    eq_rsi = rsi_strategy(prices)
    eq_macd = backtest_single(aapl, macd_signals(aapl))

    # SAVE CSVs
    prices.to_csv("outputs/adj_close.csv")
    fund.to_csv("outputs/fundamentals.csv")
    weights.to_csv("outputs/momentum_weights.csv")
    port_rets.to_csv("outputs/momentum_monthly_returns.csv")

    # SUMMARY TABLE
    summary = pd.DataFrame([
        performance_summary(eq_sma),
        performance_summary(eq_bb),
        performance_summary(eq_gv),
        performance_summary(eq_mom),
        performance_summary(eq_rsi),
        performance_summary(eq_macd)
    ], index=["SMA","Bollinger","Growth–Value","Momentum","RSI","MACD"])

    summary.to_csv("outputs/strategy_performance_summary.csv")
    print("\n===== STRATEGY PERFORMANCE SUMMARY =====")
    print(summary.round(2))

    # COMBINED PLOT
    combined = pd.DataFrame({
        "SMA": eq_sma,
        "Bollinger": eq_bb,
        "Growth-Value": eq_gv,
        "Momentum": eq_mom,
        "RSI": eq_rsi,
        "MACD": eq_macd
    }).ffill()

    combined.plot(figsize=(12,6))
    plt.title("Performance Comparison of 6 Trading Strategies (2015–2025)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.savefig("outputs/combined_strategies_equity.png", dpi=300)
    plt.show()

    print("\nAll outputs saved to folder: outputs/")

