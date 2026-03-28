"""
India Stock API Server
======================
Run this locally or deploy to Railway/Render.
Serves scraped NSE stock data as JSON with CORS enabled
so your browser widget can fetch it directly.

Usage:
    python api_server.py

Endpoints:
    GET /quote?symbol=RELIANCE          — single stock
    GET /quotes?symbols=RELIANCE,TCS    — multiple stocks
    GET /watchlist                      — all default stocks
    GET /indices                        — NIFTY 50, SENSEX, USD/INR
    GET /health                         — health check
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from scraper import scrape_stock, scrape_all, create_driver, NSE_STOCKS
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow ALL origins — lets your browser widget call this freely

# ── In-memory cache ───────────────────────────────────────────────────────────
cache = {}
cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes


def get_cached(symbol: str):
    with cache_lock:
        entry = cache.get(symbol)
        if entry and (time.time() - entry["ts"]) < CACHE_TTL:
            return entry["data"]
    return None


def set_cached(symbol: str, data: dict):
    with cache_lock:
        cache[symbol] = {"data": data, "ts": time.time()}


# ── Single stock ──────────────────────────────────────────────────────────────
@app.route("/quote")
def quote():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol parameter required"}), 400

    cached = get_cached(symbol)
    if cached:
        return jsonify({"status": "ok", "cached": True, "data": cached})

    driver = create_driver(headless=True)
    try:
        data = scrape_stock(driver, symbol)
        set_cached(symbol, data)
        return jsonify({"status": "ok", "cached": False, "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        driver.quit()


# ── Multiple stocks ───────────────────────────────────────────────────────────
@app.route("/quote")
def quote():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol parameter required"}), 400
    cached = get_cached(symbol)
    if cached:
        return jsonify({"status": "ok", "cached": True, "data": cached})
    try:
        from scraper import scrape_stock
        data = scrape_stock(symbol)
        set_cached(symbol, data)
        return jsonify({"status": "ok", "cached": False, "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Full watchlist ────────────────────────────────────────────────────────────
@app.route("/watchlist")
def watchlist():
    results = []
    to_fetch = []

    for sym in NSE_STOCKS:
        cached = get_cached(sym)
        if cached:
            results.append(cached)
        else:
            to_fetch.append(sym)

    if to_fetch:
        driver = create_driver(headless=True)
        try:
            for sym in to_fetch:
                data = scrape_stock(driver, sym)
                set_cached(sym, data)
                results.append(data)
                time.sleep(1.5)
        finally:
            driver.quit()

    return jsonify({"status": "ok", "count": len(results), "data": results})


# ── Market indices ────────────────────────────────────────────────────────────
@app.route("/indices")
def indices():
    index_symbols = {
        "NIFTY_50": "%5ENSEI",
        "SENSEX": "%5EBSESN",
        "BANK_NIFTY": "%5ENSEBANK",
    }

    results = {}
    driver = create_driver(headless=True)
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        for name, sym in index_symbols.items():
            cached = get_cached(sym)
            if cached:
                results[name] = cached
                continue

            url = f"https://finance.yahoo.com/quote/{sym}"
            driver.get(url)
            wait = WebDriverWait(driver, 12)
            try:
                price_el = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'fin-streamer[data-field="regularMarketPrice"]')
                    )
                )
                price = price_el.text.strip()
                try:
                    chg_el = driver.find_element(
                        By.CSS_SELECTOR,
                        'fin-streamer[data-field="regularMarketChangePercent"]'
                    )
                    chg = chg_el.text.strip()
                except Exception:
                    chg = "—"

                data = {"symbol": name, "price": price, "change_pct": chg}
                set_cached(sym, data)
                results[name] = data
            except Exception as e:
                results[name] = {"symbol": name, "error": str(e)}
            time.sleep(1.5)
    finally:
        driver.quit()

    # Also fetch USD/INR
    try:
        import urllib.request
        with urllib.request.urlopen(
            "https://query1.finance.yahoo.com/v8/finance/chart/USDINR=X?interval=1d&range=1d",
            timeout=5
        ) as resp:
            import json
            jdata = json.loads(resp.read())
            rate = jdata["chart"]["result"][0]["meta"]["regularMarketPrice"]
            results["USD_INR"] = {"symbol": "USD_INR", "price": str(round(rate, 2))}
    except Exception:
        results["USD_INR"] = {"symbol": "USD_INR", "price": "—"}

    return jsonify({"status": "ok", "data": results})


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "India Stock API",
        "cached_symbols": len(cache),
        "timestamp": datetime.now().isoformat()
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 India Stock API running at http://localhost:5000")
    print("   Endpoints:")
    print("   GET /quote?symbol=RELIANCE")
    print("   GET /quotes?symbols=RELIANCE,TCS,INFY")
    print("   GET /watchlist")
    print("   GET /indices")
    print("   GET /health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
