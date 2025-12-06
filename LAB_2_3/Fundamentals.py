# -------------------------
#  Fundamentals (basic)
# -------------------------
def fetch_basic_fundamentals(tickers):
    rows=[]
    for t in tickers:
        tk = yf.Ticker(t)
        try:
            info = tk.info
        except Exception:
            info = {}
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
    df = pd.DataFrame(rows).set_index("ticker")
    return df

def classify_growth_value(fund, pe_thresh=20.0):
    df = fund.copy()
    df["epsTTM"] = pd.to_numeric(df["epsTTM"], errors="coerce")
    df["trailingPE"] = pd.to_numeric(df["trailingPE"], errors="coerce")
    df["growth_flag"] = (df["epsTTM"] > 0) & (df["trailingPE"] >= pe_thresh)
    df["value_flag"] = (df["trailingPE"] < pe_thresh) & (df["epsTTM"] > 0)
    df["tag"] = "neutral"
    df.loc[df["growth_flag"], "tag"] = "growth"
    df.loc[df["value_flag"], "tag"] = "value"
    return df
