"""
India Stock Scraper — Yahoo Finance (NSE)
Uses Selenium to scrape live stock data exactly like the tutorial.
NSE stocks use .NS suffix on Yahoo Finance (e.g. RELIANCE.NS)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import json
from datetime import datetime


# ── Full Nifty 50 stocks ─────────────────────────────────────────────────────
NSE_STOCKS = [
    # Financial Services
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN",
    # IT
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    # Oil & Gas / Energy
    "RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID", "NTPC",
    "ADANIGREEN", "ADANIPORTS", "ADANIENT", "TATAPOWER",
    # Consumer / FMCG
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    # Auto
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    # Pharma / Healthcare
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    # Metals & Mining
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
    # Infrastructure / Construction
    "LT", "ULTRACEMCO", "GRASIM",
    # Telecom
    "BHARTIARTL",
    # Others
    "ASIANPAINT", "TITAN", "INDUSINDBK",
]


# Yahoo Finance uses .NS suffix for NSE stocks
def nse_to_yahoo(symbol: str) -> str:
    return f"{symbol}.NS"


# ── Setup Selenium ────────────────────────────────────────────────────────────
def create_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")

    import os
    chrome_bin = os.environ.get("CHROME_BIN", None)
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", None)

    if chrome_bin:
        options.binary_location = chrome_bin

    if chromedriver_path:
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
    else:
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    return driver


# ── Scrape a single stock ─────────────────────────────────────────────────────
def scrape_stock(driver: webdriver.Chrome, symbol: str) -> dict:
    yahoo_sym = nse_to_yahoo(symbol)
    url = f"https://finance.yahoo.com/quote/{yahoo_sym}"
    print(f"  -> Fetching {symbol} from {url}")
    driver.get(url)

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
        time.sleep(3)

        # ── Price ─────────────────────────────────────────────────────────
        price_selectors = [
            'fin-streamer[data-field="regularMarketPrice"]',
            'span[data-testid="qsp-price"]',
            '[data-field="regularMarketPrice"]',
            f'fin-streamer[data-symbol="{yahoo_sym}"][data-field="regularMarketPrice"]',
        ]
        for sel in price_selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                if el.text.strip():
                    result["price"] = el.text.strip()
                    break
            except Exception:
                continue

        # ── Change (absolute) — filtered by symbol to avoid wrong element ─
        change_selectors = [
            'span[data-testid="qsp-price-change"]',
            f'fin-streamer[data-field="regularMarketChange"][data-symbol="{yahoo_sym}"]',
        ]
        for sel in change_selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                if el.text.strip():
                    result["change"] = el.text.strip()
                    break
            except Exception:
                continue

        # ── Change percent ────────────────────────────────────────────────
        chgpct_selectors = [
            'span[data-testid="qsp-price-change-percent"]',
            f'fin-streamer[data-field="regularMarketChangePercent"][data-symbol="{yahoo_sym}"]',
        ]
        for sel in chgpct_selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                if el.text.strip():
                    result["change_pct"] = el.text.strip()
                    break
            except Exception:
                continue

        time.sleep(1)

        # ── Method 1: data-testid stats (newer Yahoo layout) ──────────────
        stat_map = {
            "PREV_CLOSE-value": "prev_close",
            "OPEN-value": "open",
            "DAYS_RANGE-value": "day_range",
            "FIFTY_TWO_WK_RANGE-value": "week52_range",
            "TD_VOLUME-value": "volume",
            "AVERAGE_VOLUME_3MONTH-value": "avg_volume",
            "MARKET_CAP-value": "market_cap",
            "PE_RATIO-value": "pe_ratio",
            "EPS_RATIO-value": "eps",
            "DIVIDEND_AND_YIELD-value": "dividend_yield",
        }
        for testid, field in stat_map.items():
            try:
                el = driver.find_element(By.CSS_SELECTOR, f'[data-testid="{testid}"]')
                val = el.text.strip()
                if val and val != "N/A":
                    if field == "day_range":
                        parts = val.split(" - ")
                        if len(parts) == 2:
                            result["day_low"] = parts[0].strip()
                            result["day_high"] = parts[1].strip()
                    elif field == "week52_range":
                        parts = val.split(" - ")
                        if len(parts) == 2:
                            result["week52_low"] = parts[0].strip()
                            result["week52_high"] = parts[1].strip()
                    else:
                        result[field] = val
            except Exception:
                continue

        # ── Method 2: table cells ─────────────────────────────────────────
        if not result["prev_close"]:
            try:
                all_tds = driver.find_elements(By.CSS_SELECTOR, "td")
                for i in range(0, len(all_tds) - 1, 2):
                    try:
                        label = all_tds[i].text.strip().lower()
                        value = all_tds[i+1].text.strip()
                        if not value or value == "N/A":
                            continue
                        if "prev" in label and "close" in label:
                            result["prev_close"] = value
                        elif label == "open":
                            result["open"] = value
                        elif "day" in label and "range" in label:
                            parts = value.split(" - ")
                            if len(parts) == 2:
                                result["day_low"] = parts[0].strip()
                                result["day_high"] = parts[1].strip()
                        elif "52" in label and "range" in label:
                            parts = value.split(" - ")
                            if len(parts) == 2:
                                result["week52_low"] = parts[0].strip()
                                result["week52_high"] = parts[1].strip()
                        elif "volume" in label and "avg" not in label:
                            result["volume"] = value
                        elif "avg" in label and "volume" in label:
                            result["avg_volume"] = value
                        elif "market cap" in label:
                            result["market_cap"] = value
                        elif "p/e" in label or "pe ratio" in label:
                            result["pe_ratio"] = value
                        elif label == "eps":
                            result["eps"] = value
                        elif "yield" in label:
                            result["dividend_yield"] = value
                    except Exception:
                        continue
            except Exception:
                pass

        # ── Method 3: list items (newer Yahoo layout) ─────────────────────
        if not result["prev_close"]:
            try:
                items = driver.find_elements(By.CSS_SELECTOR, "li")
                for item in items:
                    try:
                        text = item.text.strip()
                        if "\n" in text:
                            parts = text.split("\n")
                            label = parts[0].strip().lower()
                            value = parts[1].strip() if len(parts) > 1 else ""
                            if not value or value == "N/A":
                                continue
                            if "previous close" in label:
                                result["prev_close"] = value
                            elif label == "open":
                                result["open"] = value
                            elif "day's range" in label:
                                p = value.split(" - ")
                                if len(p) == 2:
                                    result["day_low"] = p[0].strip()
                                    result["day_high"] = p[1].strip()
                            elif "52 week" in label:
                                p = value.split(" - ")
                                if len(p) == 2:
                                    result["week52_low"] = p[0].strip()
                                    result["week52_high"] = p[1].strip()
                            elif "volume" in label and "avg" not in label:
                                result["volume"] = value
                            elif "market cap" in label:
                                result["market_cap"] = value
                            elif "p/e" in label or "pe" in label:
                                result["pe_ratio"] = value
                            elif "eps" in label:
                                result["eps"] = value
                            elif "yield" in label:
                                result["dividend_yield"] = value
                    except Exception:
                        continue
            except Exception:
                pass

    except Exception as e:
        result["error"] = str(e)
        print(f"    ERROR scraping {symbol}: {e}")

    return result


# ── Clean numeric strings ─────────────────────────────────────────────────────
def clean_number(val: str):
    if not val or val in ("-", "N/A", ""):
        return None
    cleaned = val.replace(",", "").replace("₹", "").replace("$", "")
    cleaned = cleaned.replace("(", "-").replace(")", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ── Scrape all stocks ─────────────────────────────────────────────────────────
def scrape_all(symbols: list, headless: bool = True, delay: float = 2.0) -> pd.DataFrame:
    driver = create_driver(headless=headless)
    results = []
    try:
        print(f"\nScraping {len(symbols)} NSE stocks from Yahoo Finance...\n")
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol}")
            data = scrape_stock(driver, symbol)
            results.append(data)
            if i < len(symbols):
                time.sleep(delay)
    finally:
        driver.quit()

    df = pd.DataFrame(results)
    numeric_cols = ["price", "change", "prev_close", "open",
                    "day_low", "day_high", "week52_low", "week52_high", "pe_ratio", "eps"]
    for col in numeric_cols:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(clean_number)
    return df


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, filename_prefix: str = "nse_stocks"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{filename_prefix}_{timestamp}.csv"
    json_path = f"{filename_prefix}_{timestamp}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\nSaved:\n  CSV  -> {csv_path}\n  JSON -> {json_path}")
    return csv_path, json_path


# ── Quick display ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(f"{'SYMBOL':<14} {'PRICE':<14} {'CHANGE':<14} {'MKT CAP':<16} {'P/E'}")
    print("-" * 70)
    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        price = row.get("price") or "-"
        change = row.get("change") or "-"
        chg_pct = row.get("change_pct") or ""
        mcap = row.get("market_cap") or "-"
        pe = row.get("pe_ratio") or "-"
        err = row.get("error")
        if err:
            print(f"{sym:<14} ERROR: {str(err)[:50]}")
        else:
            print(f"{sym:<14} {price:<14} {change} {chg_pct:<10} {mcap:<16} {pe}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = scrape_all(NSE_STOCKS, headless=True, delay=2.0)
    print_summary(df)
    save_results(df)
