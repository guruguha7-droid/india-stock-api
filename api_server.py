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
        try:
            s = str(val).replace(',','').replace('%','').replace('₹','')
            s = s.replace('T','').replace('B','').replace('Cr','').strip()
            return float(s)
        except: return default

    price     = n(d.get('price'))
    pe        = n(d.get('pe_ratio'))
    eps       = n(d.get('eps'))
    high52    = n(d.get('week52_high'))
    low52     = n(d.get('week52_low'))
    div_yield = n(d.get('dividend_yield'))
    chg_pct   = n(d.get('change_pct'))
    mcap_str  = str(d.get('market_cap',''))

    mcap_t = None
    try:
        val = float(mcap_str.replace('₹','').replace(',','').replace('T','').replace('B','').replace('Cr','').strip())
        if 'T' in mcap_str:    mcap_t = val
        elif 'B' in mcap_str:  mcap_t = val / 1000
        elif 'Cr' in mcap_str: mcap_t = val / 100000
    except: pass

    momentum = None
    if high52 and low52 and price and high52 > low52:
        momentum = (price - low52) / (high52 - low52)

    if focus == 'value':
        if pe:
            if pe < 10:        score += 30; reasons.append(f'Very cheap P/E {pe:.1f}')
            elif pe < 15:      score += 25; reasons.append(f'Cheap P/E {pe:.1f}')
            elif pe < 20:      score += 18; reasons.append(f'Fair P/E {pe:.1f}')
            elif pe < 28:      score += 10
            elif pe < 40:      score += 3
            else:              score -= 8;  reasons.append(f'Expensive P/E {pe:.1f}')
        if momentum is not None:
            if momentum < 0.25:   score += 20; reasons.append('Deep discount to 52W high')
            elif momentum < 0.45: score += 14; reasons.append('Trading at discount')
            elif momentum < 0.65: score += 7
            elif momentum > 0.85: score -= 5
        if div_yield:
            if div_yield > 3:   score += 15; reasons.append(f'Good dividend {div_yield:.1f}%')
            elif div_yield > 1: score += 8

    elif focus == 'growth':
        if eps:
            if eps > 150:      score += 30; reasons.append(f'Very high EPS ₹{eps:.0f}')
            elif eps > 80:     score += 24; reasons.append(f'High EPS ₹{eps:.0f}')
            elif eps > 40:     score += 18; reasons.append(f'Good EPS ₹{eps:.0f}')
            elif eps > 15:     score += 12; reasons.append(f'Moderate EPS ₹{eps:.0f}')
            elif eps > 0:      score += 5
            else:              score -= 10; reasons.append('Negative EPS — unprofitable')
        if momentum is not None:
            if momentum > 0.75:   score += 25; reasons.append('Strong price momentum')
            elif momentum > 0.55: score += 18; reasons.append('Good momentum')
            elif momentum > 0.35: score += 10
            elif momentum < 0.2:  score -= 8;  reasons.append('Weak momentum')
        if pe:
            if 15 <= pe <= 35:  score += 12; reasons.append('Healthy growth P/E')
            elif pe < 15:       score += 8
            elif pe > 60:       score -= 8;  reasons.append('Very expensive')

    elif focus == 'dividend':
        if div_yield:
            if div_yield > 6:    score += 35; reasons.append(f'Very high dividend {div_yield:.1f}%')
            elif div_yield > 4:  score += 28; reasons.append(f'High dividend {div_yield:.1f}%')
            elif div_yield > 2:  score += 18; reasons.append(f'Good dividend {div_yield:.1f}%')
            elif div_yield > 1:  score += 8;  reasons.append(f'Moderate dividend {div_yield:.1f}%')
            else:                score -= 5;  reasons.append('Very low dividend')
        else:
            score -= 15; reasons.append('No dividend data')
        if pe:
            if pe < 12:   score += 20; reasons.append(f'Very cheap P/E {pe:.1f}')
            elif pe < 18: score += 14
            elif pe < 25: score += 7
            elif pe > 40: score -= 8
        if momentum is not None:
            if momentum > 0.3: score += 8
            else: score -= 5

    elif focus == 'momentum':
        if momentum is not None:
            if momentum > 0.85:   score += 40; reasons.append('Near 52W high — very strong momentum')
            elif momentum > 0.70: score += 32; reasons.append('Strong momentum')
            elif momentum > 0.55: score += 22; reasons.append('Good momentum')
            elif momentum > 0.40: score += 12
            elif momentum > 0.25: score += 4
            else:                 score -= 10; reasons.append('Weak — near 52W low')
        if chg_pct:
            if chg_pct > 2:    score += 12; reasons.append(f'Up {chg_pct:.1f}% today')
            elif chg_pct > 0:  score += 5
            elif chg_pct < -3: score -= 8
        if eps and eps > 0:
            score += 8; reasons.append('Profitable company')

    if risk == 'conservative':
        if mcap_t:
            if mcap_t > 8:    score += 20; reasons.append('Very large cap — low risk')
            elif mcap_t > 3:  score += 12; reasons.append('Large cap — stable')
            elif mcap_t < 1:  score -= 15; reasons.append('Small cap — high risk')
        if chg_pct and abs(chg_pct) > 4:
            score -= 12; reasons.append('High volatility today')
        if pe and pe > 40:
            score -= 10
    elif risk == 'moderate':
        if mcap_t:
            if mcap_t > 5:    score += 10; reasons.append('Large cap stability')
            elif mcap_t > 2:  score += 6
            elif mcap_t < 0.5: score -= 8
        if chg_pct and abs(chg_pct) > 6:
            score -= 8
    elif risk == 'aggressive':
        if mcap_t and mcap_t < 0.5:
            score -= 5
        if momentum and momentum > 0.7:
            score += 8
        if chg_pct and chg_pct > 3:
            score += 10; reasons.append(f'Strong move today +{chg_pct:.1f}%')

    if eps and eps > 0:
        score += 5

    score = max(0, min(100, score))
    return round(score, 1), reasons[:3]


def get_risk_rating(d, focus, risk):
    base = 5

    def n(val):
        try: return float(str(val).replace(',','').replace('%','').replace('₹','').replace('T','').replace('B','').replace('Cr','').strip())
        except: return None

    pe       = n(d.get('pe_ratio'))
    chg_pct  = n(d.get('change_pct'))
    mcap_str = str(d.get('market_cap',''))

    mcap_t = None
    try:
        val = float(mcap_str.replace('₹','').replace(',','').replace('T','').replace('B','').replace('Cr','').strip())
        if 'T' in mcap_str:    mcap_t = val
        elif 'B' in mcap_str:  mcap_t = val / 1000
        elif 'Cr' in mcap_str: mcap_t = val / 100000
    except: pass

    if mcap_t:
        if mcap_t > 10:    base -= 3
        elif mcap_t > 5:   base -= 2
        elif mcap_t > 2:   base -= 1
        elif mcap_t < 0.5: base += 3
        elif mcap_t < 1:   base += 1
    if pe:
        if pe > 60:   base += 3
        elif pe > 40: base += 2
        elif pe > 25: base += 1
        elif pe < 12: base -= 1
    if chg_pct:
        if abs(chg_pct) > 5: base += 2
        elif abs(chg_pct) > 3: base += 1
    if focus == 'momentum': base += 1
    if focus == 'dividend': base -= 1
    if focus == 'value':    base -= 1
    if risk == 'conservative': base -= 1
    if risk == 'aggressive':   base += 1

    return max(1, min(10, base))


def get_moat(d):
    mcap_str = str(d.get('market_cap', ''))
    try:
        val = float(mcap_str.replace('₹','').replace(',','').replace('T','').replace('B','').replace('Cr','').strip())
        if 'T' in mcap_str:
            if val > 10: return 'Strong'
            if val > 3:  return 'Moderate'
            return 'Weak'
        elif 'B' in mcap_str:
            return 'Weak'
        elif 'Cr' in mcap_str:
            if val > 1000000: return 'Strong'
            if val > 300000:  return 'Moderate'
            return 'Weak'
    except:
        pass
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
            'risk_rating':    get_risk_rating(d, focus, risk),
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
