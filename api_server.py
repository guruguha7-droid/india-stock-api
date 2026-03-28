"""
India Stock API Server
======================
Run this locally or deploy to Railway/Render.
Serves NSE stock data as JSON with CORS enabled
so your browser widget can fetch it directly.

Usage:
    python api_server.py

Endpoints:
    GET /quote?symbol=RELIANCE          — single stock
    GET /quotes?symbols=RELIANCE,TCS    — multiple stocks
    GET /watchlist                      — all default stocks
    GET /indices                        — NIFTY 50, SENSEX, BANK NIFTY, USD/INR
    GET /health                         — health check
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from scraper import scrape_stock, scrape_all, NSE_STOCKS
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

    try:
        data = scrape_stock(symbol)
        set_cached(symbol, data)
        return jsonify({"status": "ok", "cached": False, "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Multiple stocks ───────────────────────────────────────────────────────────
@app.route("/quotes")
def quotes():
    raw = request.args.get("symbols", "")
    if not raw:
        return jsonify({"error": "symbols parameter required"}), 400

    symbols = [s.upper().strip() for s in raw.split(",") if s.strip()]
    results = []
    for sym in symbols:
        cached = get_cached(sym)
        if cached:
            results.append(cached)
        else:
            try:
                data = scrape_stock(sym)
                set_cached(sym, data)
                results.append(data)
            except Exception as e:
                results.append({"symbol": sym, "error": str(e)})

    return jsonify({"status": "ok", "count": len(results), "data": results})


# ── Full watchlist ────────────────────────────────────────────────────────────
@app.route("/watchlist")
def watchlist():
    results = []
    for sym in NSE_STOCKS:
        cached = get_cached(sym)
        if cached:
            results.append(cached)
        else:
            data = scrape_stock(sym)
            set_cached(sym, data)
            results.append(data)

    return jsonify({"status": "ok", "count": len(results), "data": results})


# ── Market indices ────────────────────────────────────────────────────────────
@app.route("/indices")
def indices():
    import yfinance as yf

    index_map = {
        "NIFTY_50":   "^NSEI",
        "SENSEX":     "^BSESN",
        "BANK_NIFTY": "^NSEBANK",
        "USD_INR":    "USDINR=X",
    }

    results = {}
    for name, sym in index_map.items():
        cached = get_cached(sym)
        if cached:
            results[name] = cached
            continue

        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

            chg_pct = None
            if price and prev_close:
                chg_pct = f"{((price - prev_close) / prev_close) * 100:+.2f}%"

            data = {
                "symbol": name,
                "price": f"{price:,.2f}" if price else None,
                "change_pct": chg_pct,
            }
            set_cached(sym, data)
            results[name] = data
        except Exception as e:
            results[name] = {"symbol": name, "error": str(e)}

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
    print("\nIndia Stock API running at http://localhost:5000")
    print("   Endpoints:")
    print("   GET /quote?symbol=RELIANCE")
    print("   GET /quotes?symbols=RELIANCE,TCS,INFY")
    print("   GET /watchlist")
    print("   GET /indices")
    print("   GET /health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
