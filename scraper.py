"""
India Stock Scraper — yfinance (no browser needed)
Fetches live NSE data from Yahoo Finance API directly.
Works perfectly on Render/Railway free tier.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


NSE_STOCKS = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    "RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID", "NTPC",
    "ADANIGREEN", "ADANIPORTS", "ADANIENT", "TATAPOWER",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
    "LT", "ULTRACEMCO", "GRASIM",
    "BHARTIARTL",
    "ASIANPAINT", "TITAN", "INDUSINDBK",
]


def nse_to_yahoo(symbol: str) -> str:
    return f"{symbol}.NS"


def scrape_stock(symbol: str) -> dict:
    """Fetch live NSE stock data using yfinance — no browser needed."""
    yahoo_sym = nse_to_yahoo(symbol)
    print(f"  -> Fetching {symbol} ({yahoo_sym})")

    result = {
        "symbol": symbol,
        "yahoo_symbol": yahoo_sym,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": None, "change": None, "change_pct": None,
        "prev_close": None, "open": None, "day_low": None,
        "day_high": None, "week52_low": None, "week52_high": None,
        "volume": None, "avg_volume": None, "market_cap": None,
        "pe_ratio": None, "eps": None, "dividend_yield": None,
        "error": None,
    }

    try:
        ticker = yf.Ticker(yahoo_sym)
        info = ticker.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        result["price"] = f"{price:,.2f}" if price else None
        result["prev_close"] = f"{prev_close:,.2f}" if prev_close else None
        result["open"] = f"{info.get('open', info.get('regularMarketOpen', 0)):,.2f}" if info.get("open") else None
        result["day_low"] = f"{info.get('dayLow', info.get('regularMarketDayLow', 0)):,.2f}" if info.get("dayLow") else None
        result["day_high"] = f"{info.get('dayHigh', info.get('regularMarketDayHigh', 0)):,.2f}" if info.get("dayHigh") else None
        result["week52_low"] = f"{info.get('fiftyTwoWeekLow', 0):,.2f}" if info.get("fiftyTwoWeekLow") else None
        result["week52_high"] = f"{info.get('fiftyTwoWeekHigh', 0):,.2f}" if info.get("fiftyTwoWeekHigh") else None
        result["volume"] = f"{info.get('volume', info.get('regularMarketVolume', 0)):,}" if info.get("volume") else None
        result["avg_volume"] = f"{info.get('averageVolume', 0):,}" if info.get("averageVolume") else None
        result["pe_ratio"] = f"{info.get('trailingPE', 0):.2f}" if info.get("trailingPE") else None
        result["eps"] = f"{info.get('trailingEps', 0):.2f}" if info.get("trailingEps") else None
        result["market_cap"] = _fmt_cap(info.get("marketCap"))
        div = info.get("dividendYield")
        if div:
            # yfinance 1.x returns dividend yield already as a percentage (e.g. 3.54)
            # older versions returned a decimal (e.g. 0.0354) — normalise both cases
            result["dividend_yield"] = f"{div:.2f}%" if div > 1 else f"{div * 100:.2f}%"

        if price and prev_close:
            chg = price - prev_close
            chg_pct = (chg / prev_close) * 100
            result["change"] = f"{chg:+.2f}"
            result["change_pct"] = f"({chg_pct:+.2f}%)"

    except Exception as e:
        result["error"] = str(e)
        print(f"    ERROR: {e}")

    return result


def _fmt_cap(val):
    if not val:
        return None
    if val >= 1e12:
        return f"₹{val/1e12:.2f}T"
    if val >= 1e9:
        return f"₹{val/1e9:.2f}B"
    if val >= 1e7:
        return f"₹{val/1e7:.2f}Cr"
    return f"₹{val:,.0f}"


def scrape_all(symbols=None, **kwargs):
    symbols = symbols or NSE_STOCKS
    results = []
    print(f"\nFetching {len(symbols)} NSE stocks via yfinance...\n")
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        results.append(scrape_stock(sym))
    return pd.DataFrame(results)


def create_driver(**kwargs):
    """Kept for compatibility — not used with yfinance."""
    return None


if __name__ == "__main__":
    df = scrape_all(NSE_STOCKS[:5])
    print(df[["symbol", "price", "change", "change_pct", "market_cap", "pe_ratio"]])