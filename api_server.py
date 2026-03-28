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


# ── Custom Screening Engine ───────────────────────────────────────────────────

def score_stock(d, focus, risk):
    score = 0
    reasons = []

    def n(val, default=None):
        if val is None: return default
        try: return float(str(val).replace(',','').replace('%','').replace('₹','').replace('T','').replace('B','').replace('Cr',''))
        except: return default

    price     = n(d.get('price'))
    pe        = n(d.get('pe_ratio'))
    eps       = n(d.get('eps'))
    high52    = n(d.get('week52_high'))
    low52     = n(d.get('week52_low'))
    div_yield = n(d.get('dividend_yield'))
    mktcap    = n(d.get('market_cap'))
    chg_pct   = n(d.get('change_pct'))

    if focus in ['value', 'dividend']:
        if pe:
            if pe < 12:   score += 25; reasons.append('Very cheap P/E')
            elif pe < 18: score += 20; reasons.append('Attractive P/E')
            elif pe < 25: score += 12; reasons.append('Fair P/E')
            elif pe < 35: score += 5
            else:         score -= 5;  reasons.append('Expensive P/E')

    if focus in ['growth', 'momentum']:
        if eps:
            if eps > 100:  score += 20; reasons.append('High EPS')
            elif eps > 50: score += 15; reasons.append('Good EPS')
            elif eps > 20: score += 10
            elif eps > 0:  score += 5

    if high52 and low52 and price and high52 != low52:
        momentum = (price - low52) / (high52 - low52)
        if focus == 'momentum':
            if momentum > 0.8:   score += 25; reasons.append('Near 52W high — strong momentum')
            elif momentum > 0.6: score += 18
            elif momentum > 0.4: score += 10
            elif momentum > 0.2: score += 5
            else:                score -= 5;  reasons.append('Near 52W low')
        elif focus == 'value':
            if momentum < 0.3:   score += 20; reasons.append('Trading at discount to 52W high')
            elif momentum < 0.5: score += 12
            elif momentum < 0.7: score += 6

    if focus == 'dividend':
        if div_yield:
            if div_yield > 4:    score += 30; reasons.append(f'High dividend {div_yield:.1f}%')
            elif div_yield > 2:  score += 20; reasons.append(f'Good dividend {div_yield:.1f}%')
            elif div_yield > 1:  score += 10; reasons.append(f'Moderate dividend {div_yield:.1f}%')
            else:                score += 3
    elif div_yield and div_yield > 1:
        score += 5

    if risk == 'conservative':
        if mktcap:
            if mktcap > 10:   score += 15; reasons.append('Large cap — lower risk')
            elif mktcap > 5:  score += 8
            elif mktcap < 1:  score -= 10; reasons.append('Small cap — higher risk')
        if chg_pct and chg_pct < -3:
            score -= 10; reasons.append('High volatility today')
    elif risk == 'aggressive':
        if chg_pct and chg_pct > 1:
            score += 10; reasons.append('Positive momentum today')
        if pe and pe > 30:
            score += 5
    elif risk == 'moderate':
        if mktcap and mktcap > 5:
            score += 8; reasons.append('Mid-large cap stability')

    if eps and eps > 0:
        score += 5; reasons.append('Profitable company')

    score = max(0, min(100, score))
    return score, reasons[:3]


def get_risk_rating(score):
    if score >= 75: return 3
    if score >= 55: return 5
    if score >= 35: return 7
    return 9


def get_moat(d):
    mktcap = None
    try: mktcap = float(str(d.get('market_cap','')).replace(',','').replace('T','').replace('B','').replace('Cr',''))
    except: pass
    if mktcap and mktcap > 15: return 'Strong'
    if mktcap and mktcap > 5:  return 'Moderate'
    return 'Weak'


@app.route("/screen")
def screen():
    risk   = request.args.get('risk', 'moderate').lower()
    focus  = request.args.get('focus', 'value').lower()
    sector = request.args.get('sector', 'any').lower()
    amount = request.args.get('amount', '500000')

    all_data = []
    for sym in NSE_STOCKS:
        cached = get_cached(sym)
        if cached:
            all_data.append(cached)
        else:
            try:
                d = scrape_stock(sym)
                set_cached(sym, d)
                all_data.append(d)
            except:
                pass

    if sector != 'any':
        sector_map = {
            'it': ['technology','information technology','it services','software'],
            'banking': ['banking','financial services','finance','bank'],
            'fmcg': ['consumer','fmcg','consumer staples','consumer defensive'],
            'pharma': ['pharma','healthcare','health care','drug'],
            'auto': ['auto','automobile','automotive'],
            'energy': ['energy','oil','gas','petroleum'],
            'infrastructure': ['infrastructure','construction','industrial'],
            'metals': ['metals','mining','steel','aluminium'],
            'chemicals': ['chemicals','specialty chemicals'],
        }
        keywords = sector_map.get(sector, [sector])
        filtered = [d for d in all_data if any(
            kw in str(d.get('sector','')).lower() or
            kw in str(d.get('industry','')).lower()
            for kw in keywords
        )]
        if len(filtered) >= 3:
            all_data = filtered

    scored = []
    for d in all_data:
        if d.get('error') or not d.get('price'):
            continue
        score, reasons = score_stock(d, focus, risk)
        scored.append({
            'symbol':         d.get('symbol'),
            'price':          d.get('price'),
            'change':         d.get('change'),
            'change_pct':     d.get('change_pct'),
            'pe_ratio':       d.get('pe_ratio'),
            'market_cap':     d.get('market_cap'),
            'dividend_yield': d.get('dividend_yield'),
            'week52_high':    d.get('week52_high'),
            'week52_low':     d.get('week52_low'),
            'eps':            d.get('eps'),
            'sector':         d.get('sector'),
            'score':          round(score, 1),
            'reasons':        reasons,
            'moat':           get_moat(d),
            'risk_rating':    get_risk_rating(score),
        })

    scored.sort(key=lambda x: x['score'], reverse=True)
    top10 = scored[:10]

    try:
        total = float(str(amount).replace(',',''))
    except:
        total = 500000

    total_score = sum(s['score'] for s in top10) or 1
    for s in top10:
        weight = s['score'] / total_score
        s['allocation'] = f"₹{total * weight:,.0f}"
        s['allocation_pct'] = f"{weight*100:.1f}%"

    return jsonify({
        'status': 'ok',
        'profile': {'risk': risk, 'focus': focus, 'sector': sector, 'amount': amount},
        'total_screened': len(scored),
        'top10': top10
    })


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
