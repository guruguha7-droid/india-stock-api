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

import os
import json as _json
import logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
from flask import Flask, jsonify, request
from flask_cors import CORS
from scraper import scrape_stock, NSE_STOCKS
from nse_scraper import get_quote as nse_quote, get_indices as nse_indices, get_session
try:
    from ml_screener import run_ml_screen
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    print(f"ML screener not available: {e}")
import threading
import time
from datetime import datetime

# ── Disk cache helpers — survive process restarts ─────────────────────────────
DISK_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Market constants — update manually when RBI changes rates ────────────────
RBI_REPO_RATE = 6.5   # Current RBI repo rate — update when changed

def save_disk_cache(name: str, data):
    try:
        path = os.path.join(DISK_CACHE_DIR, f'_{name}_cache.json')
        with open(path, 'w') as f:
            _json.dump({'data': data, 'ts': time.time()}, f)
    except Exception as e:
        print(f"  Disk cache save error ({name}): {e}")

def load_disk_cache(name: str, max_age_hours: int = 24):
    try:
        path = os.path.join(DISK_CACHE_DIR, f'_{name}_cache.json')
        if os.path.exists(path):
            with open(path) as f:
                entry = _json.load(f)
            if (time.time() - entry.get('ts', 0)) / 3600 < max_age_hours:
                return entry['data']
    except Exception as e:
        print(f"  Disk cache load error ({name}): {e}")
    return None

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


# ── Nifty benchmark cache — refresh every 6 hours ────────────────────────────
_nifty_cache = {'data': None, 'ts': 0}
NIFTY_TTL = 21600


def get_nifty_close():
    """Get Nifty benchmark data, cached 6 hours."""
    import yfinance as _yf
    import pandas as pd
    now = time.time()
    if _nifty_cache['data'] is not None and (now - _nifty_cache['ts']) < NIFTY_TTL:
        return _nifty_cache['data']
    try:
        df = _yf.download("^NSEI", period="2y", interval="1d",
                          auto_adjust=True, progress=False)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        close = pd.Series(df['Close'].squeeze().values,
                          index=df.index, dtype=float)
        _nifty_cache['data'] = close
        _nifty_cache['ts']   = now
        return close
    except Exception:
        return _nifty_cache['data']  # return stale if download fails


# ── Nightly precomputed cache ─────────────────────────────────────────────────
_nightly_cache = {'data': None, 'ts': 0}
NIGHTLY_CACHE_TTL = 3600  # reload from disk every hour


def get_nightly_cache():
    """Load nightly_cache.json, cached in memory 1 hour."""
    import json
    now = time.time()
    if _nightly_cache['data'] and (now - _nightly_cache['ts']) < NIGHTLY_CACHE_TTL:
        return _nightly_cache['data']
    try:
        path = os.path.join(os.path.dirname(__file__), 'nightly_cache.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            _nightly_cache['data'] = data
            _nightly_cache['ts']   = now
            print(f"  Nightly cache loaded — {len(data.get('stocks',{}))} stocks, built {data.get('built_at','?')[:16]}")
            return data
    except Exception as e:
        print(f"  Nightly cache load error: {e}")
    return None


# ── Per-stock ML features cache — 5 minutes ──────────────────────────────────
_ml_features_cache = {}
ML_FEATURES_TTL = 300


def get_stock_features_cached(symbol, nifty_close):
    """Get ML features for a stock, cached 5 minutes."""
    from ml_screener import get_features
    now = time.time()
    if symbol in _ml_features_cache:
        entry = _ml_features_cache[symbol]
        if (now - entry['ts']) < ML_FEATURES_TTL:
            return entry['data']
    f = get_features(symbol, nifty_close)
    if f:
        _ml_features_cache[symbol] = {'data': f, 'ts': now}
    return f


# ── Single stock ──────────────────────────────────────────────────────────────
@app.route("/quote")
def quote():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    try:
        data = nse_quote(symbol)
        return jsonify({"status": "ok", "data": data})
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
        sector_symbols = {
            'it': ['TCS','INFY','WIPRO','HCLTECH','TECHM','LTIM',
                'PERSISTENT','MPHASIS','COFORGE','KPITTECH'],
            'banking': ['HDFCBANK','ICICIBANK','SBIN','AXISBANK','KOTAKBANK',
                        'INDUSINDBK','BAJFINANCE','BAJAJFINSV','SHRIRAMFIN',
                        'BANKBARODA','PNB','CANBK','CHOLAFIN','MUTHOOTFIN','MANAPPURAM'],
            'fmcg': ['HINDUNILVR','ITC','NESTLEIND','BRITANNIA','TATACONSUM',
                    'MARICO','DABUR','COLPAL','GODREJCP','EMAMILTD'],
            'pharma': ['SUNPHARMA','DRREDDY','CIPLA','DIVISLAB','APOLLOHOSP',
                    'TORNTPHARM','LUPIN','AUROPHARMA','ALKEM'],
            'auto': ['MARUTI','M&M','BAJAJ-AUTO','HEROMOTOCO','EICHERMOT',
                    'TVSMOTOR','MOTHERSON','BALKRISIND','APOLLOTYRE'],
            'energy': ['RELIANCE','ONGC','BPCL','IOC','TATAPOWER',
                    'ADANIGREEN','NTPC','POWERGRID','COALINDIA'],
            'infrastructure': ['LT','ULTRACEMCO','GRASIM','ADANIPORTS',
                            'ADANIENT','SIEMENS','ABB','CUMMINSIND',
                            'HAVELLS','DLF','OBEROIRLTY','RAMCOCEM'],
            'metals': ['TATASTEEL','JSWSTEEL','HINDALCO','VEDL','COALINDIA'],
            'chemicals': ['DIVISLAB','CIPLA','AUROPHARMA'],
        }
        allowed = sector_symbols.get(sector.lower(), [])
        if allowed:
            filtered = [d for d in all_data if d.get('symbol','') in allowed]
            if len(filtered) >= 2:
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
    scored = [s for s in scored if s['score'] > 0]
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
    try:
        data = nse_indices()
        result = {}

        mapping = {
            'NIFTY 50':   'NIFTY_50',
            'NIFTY BANK': 'NIFTY_BANK',
            'SENSEX':     'SENSEX',
            'USD_INR':    'USD_INR',
            'INDIA VIX':  'INDIA_VIX',
        }

        for nse_name, key in mapping.items():
            d = data.get(nse_name)
            if d:
                chg = d.get('change_pct')
                result[key] = {
                    'price':      d.get('price'),
                    'change_pct': f"{float(chg):+.2f}%" if chg else '—',
                }

        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/debug-valuation")
def debug_valuation():
    import yfinance as yf
    t  = yf.Ticker("RELIANCE.NS")
    fi = t.fast_info
    attrs = {a: str(getattr(fi, a, 'MISSING')) for a in dir(fi) if not a.startswith('_')}
    return jsonify(attrs)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "India Stock API",
        "cached_symbols": len(cache),
        "timestamp": datetime.now().isoformat()
    })


# ── ML Screener ───────────────────────────────────────────────────────────────
@app.route("/ml-screen")
def ml_screen():
    if not ML_AVAILABLE:
        return jsonify({"status": "error",
                        "message": "ML libraries not installed"}), 503
    try:
        import math
        top_n = int(request.args.get('top', 10))
        result = run_ml_screen(top_n=top_n)

        # Fix NaN values before JSON serialization
        def fix_nan(obj):
            if isinstance(obj, dict):
                return {k: fix_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_nan(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            return obj

        result = fix_nan(result)
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Dividend yield normaliser ─────────────────────────────────────────────────
def _safe_div_yield(div_yield_raw, div_rate=None, price=None):
    """
    yfinance 'dividendYield' is inconsistent:
      - Sometimes returns a fraction already: 0.0175  (= 1.75%)
      - Sometimes returns a percentage:       1.75    (= 1.75%)
    We normalise to a clean decimal fraction so the frontend can just do * 100.
    """
    if div_yield_raw is not None:
        val = float(div_yield_raw)
        # If > 1.0 it's already a percentage — convert to fraction
        if val > 1.0:
            val = val / 100.0
        return round(val, 6)
    # Fallback: compute from dividendRate / price
    if div_rate and price:
        try:
            return round(float(div_rate) / float(price), 6)
        except Exception:
            pass
    return None


# ── Quick quote — price + valuation + sentiment, no ML/Screener (~3s) ────────
@app.route("/quick-quote")
def quick_quote():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    import math
    result = {"symbol": symbol}

    def fetch_q():
        try:
            result["quote"] = nse_quote(symbol)
        except Exception:
            result["quote"] = {}

    def fetch_val():
        try:
            import yfinance as yf
            import numpy as np
            info = yf.Ticker(f"{symbol}.NS").info
            rsi_val = None
            try:
                df = yf.download(f"{symbol}.NS", period="1mo", interval="1d",
                                 auto_adjust=True, progress=False)
                if len(df) >= 15:
                    if hasattr(df.columns, 'levels'):
                        df.columns = df.columns.get_level_values(0)
                    closes = df['Close'].squeeze().values.astype(float)
                    d = np.diff(closes[-16:])
                    g = d[d > 0].mean() if len(d[d > 0]) > 0 else 0.001
                    l = abs(d[d < 0].mean()) if len(d[d < 0]) > 0 else 0.001
                    rsi_val = round(float(100 - 100 / (1 + g / l)), 1)
            except Exception:
                pass
            dy = _safe_div_yield(info.get('dividendYield'),
                                 info.get('dividendRate'),
                                 info.get('currentPrice'))
            result["valuation"] = {
                "pe_ratio":       info.get('trailingPE'),
                "eps":            info.get('trailingEps'),
                "dividend_yield": dy,
                "profit_margin":  info.get('profitMargins'),
                "revenue_growth": info.get('revenueGrowth'),
            }
            result["rsi"] = rsi_val
        except Exception:
            result["valuation"] = {}

    def fetch_sent():
        try:
            from news_sentiment import get_sentiment_score
            result["sentiment"] = get_sentiment_score(symbol)
        except Exception:
            result["sentiment"] = {"sentiment_score": 0, "sentiment_label": "neutral"}

    threads = [
        threading.Thread(target=fetch_q),
        threading.Thread(target=fetch_val),
        threading.Thread(target=fetch_sent),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)

    def fix_nan(obj):
        if isinstance(obj, dict):   return {k: fix_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [fix_nan(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj): return None
        return obj

    return jsonify(fix_nan({"status": "ok", **result}))


# ── Single-stock deep analysis ────────────────────────────────────────────────
@app.route("/stock-analysis")
def stock_analysis():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    import math
    import threading
    from news_sentiment import get_sentiment_score
    from macro_sentiment import apply_macro_to_stock

    result = {"symbol": symbol, "status": "ok"}

    # ── Run all data fetches in parallel ─────────────────────────────
    def fetch_quote():
        try:
            result["quote"] = nse_quote(symbol)
        except Exception:
            result["quote"] = {}

    def fetch_ml():
        # ── Try nightly cache first ───────────────────────────────────
        try:
            nc = get_nightly_cache()
            if nc and symbol in nc.get('stocks', {}):
                cached = nc['stocks'][symbol]
                result['ml']            = cached['ml']
                result['_val_from_cache'] = cached.get('valuation')
                result['chart_insights']  = cached.get('chart_insights', {})
                return
        except Exception:
            pass

        # ── Fallback: compute live ────────────────────────────────────
        try:
            import joblib
            import pandas as pd
            import yfinance as _yf
            import numpy as np

            saved    = joblib.load(os.path.join(os.path.dirname(__file__), 'ml_model.pkl'))
            model    = saved['model']
            features = saved['features']
            accuracy = saved['accuracy']

            # Patch sklearn version mismatch
            try:
                steps = model.steps if hasattr(model, 'steps') else []
                for _, step in steps:
                    if hasattr(step, 'statistics_') and not hasattr(step, '_fill_dtype'):
                        step._fill_dtype = step.statistics_.dtype
            except Exception:
                pass

            stock_df = _yf.download(f"{symbol}.NS", period="2y", interval="1d",
                                    auto_adjust=True, progress=False)
            nifty_close_cached = get_nifty_close()
            if nifty_close_cached is not None and len(nifty_close_cached) >= 100:
                import pandas as _pd
                nifty_df = _pd.DataFrame({'Close': nifty_close_cached})
                nifty_df.index = nifty_close_cached.index
            else:
                nifty_df = _yf.download("^NSEI", period="2y", interval="1d",
                                        auto_adjust=True, progress=False)

            if stock_df is None or len(stock_df) < 100:
                result["ml"] = {"error": "Insufficient price history"}
                return

            if hasattr(stock_df.columns, 'levels'):
                stock_df.columns = stock_df.columns.get_level_values(0)
            if hasattr(nifty_df.columns, 'levels'):
                nifty_df.columns = nifty_df.columns.get_level_values(0)

            sw_s = pd.Series(stock_df['Close'].squeeze().values,
                             index=stock_df.index, dtype=float)
            nw = None
            if nifty_df is not None and len(nifty_df) >= 100:
                nifty_close_col = nifty_df['Close']
                if hasattr(nifty_close_col, 'columns'):
                    nifty_close_col = nifty_close_col.iloc[:, 0]
                nw_s = pd.Series(nifty_close_col.squeeze().values,
                                 index=nifty_df.index, dtype=float)
                common = sw_s.index.intersection(nw_s.index)
                if len(common) >= 100:
                    sw = sw_s.loc[common].values.astype(float)
                    nw = nw_s.loc[common].values.astype(float)
                else:
                    sw = sw_s.values.astype(float)
            else:
                sw = sw_s.values.astype(float)

            cp = float(sw[-1])
            if cp <= 0 or np.isnan(cp):
                result["ml"] = {"error": "Invalid price data"}
                return

            def sr(arr, b):
                return float(arr[-1] / arr[-b] - 1) if len(arr) > b and arr[-b] > 0 else 0.0

            ret_1m = sr(sw, 22); ret_3m = sr(sw, 63)
            ret_6m = sr(sw, 126); ret_1y = sr(sw, min(200, len(sw) - 1))
            rs_1m  = ret_1m - sr(nw, 22) if nw is not None else 0.0
            rs_3m  = ret_3m - sr(nw, 63) if nw is not None else 0.0

            n50  = min(50,  len(sw)); ma50  = float(np.mean(sw[-n50:]))
            n200 = min(200, len(sw)); ma200 = float(np.mean(sw[-n200:]))

            dr = np.diff(sw) / sw[:-1]; dr = dr[~np.isnan(dr)]
            vol_1m = float(np.std(dr[-22:]) * np.sqrt(252)) if len(dr) >= 22 else 0.3
            vol_3m = float(np.std(dr[-63:]) * np.sqrt(252)) if len(dr) >= 63 else 0.3

            h52 = float(np.max(sw[-252:])) if len(sw) >= 252 else float(np.max(sw))
            l52 = float(np.min(sw[-252:])) if len(sw) >= 252 else float(np.min(sw))
            rng = h52 - l52

            d_rsi = np.diff(sw[-15:]) if len(sw) >= 15 else np.array([0.001, -0.001])
            g     = d_rsi[d_rsi > 0].mean() if len(d_rsi[d_rsi > 0]) > 0 else 0.001
            ls    = abs(d_rsi[d_rsi < 0].mean()) if len(d_rsi[d_rsi < 0]) > 0 else 0.001

            f = {
                'ret_1m': ret_1m, 'ret_3m': ret_3m,
                'ret_6m': ret_6m, 'ret_1y': ret_1y,
                'rs_1m':  rs_1m,  'rs_3m':  rs_3m,
                'price_to_ma50':  cp / ma50  - 1 if ma50  > 0 else 0,
                'price_to_ma200': cp / ma200 - 1 if ma200 > 0 else 0,
                'golden_cross':   1 if ma50 > ma200 else 0,
                'vol_1m': vol_1m, 'vol_3m': vol_3m,
                'pos52':  float((cp - l52) / rng) if rng > 0 else 0.5,
                'rsi':    float(100 - 100 / (1 + g / ls)),
                'vol_trend': float(vol_1m / vol_3m) if vol_3m > 0 else 1.0,
                'roce_latest_pct': 12.0, 'opm_latest_pct':  12.0,
                'sales_cagr_5y':   10.0, 'profit_cagr_5y':   8.0,
                'eps_cagr_5y':      8.0, 'sales_growth_1y':  8.0,
                'profit_growth_1y': 8.0, 'opm_trend_5y':     0.0,
                'roce_trend_5y':    0.0, 'promoter_pct':    45.0,
                'fii_pct':         15.0, 'fcf_positive_3y':  0.5,
                'debt_reducing':    0.5, 'screener_de':     50.0,
                'pe_ratio':        22.0, 'pb_ratio':         3.0, 'peg_ratio': 2.5,
            }

            try:
                sdf     = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      'screener_fundamentals.csv'))
                csv_sym = {'LTM': 'LTIM'}.get(symbol, symbol)
                row     = sdf[sdf['symbol'] == csv_sym]
                r       = row.iloc[0].to_dict() if not row.empty else {}
                for k, default in [
                    ('roce_latest_pct', 12.0), ('opm_latest_pct',  12.0),
                    ('sales_cagr_5y',   10.0), ('profit_cagr_5y',   8.0),
                    ('eps_cagr_5y',      8.0), ('sales_growth_1y',  8.0),
                    ('profit_growth_1y', 8.0), ('opm_trend_5y',     0.0),
                    ('roce_trend_5y',    0.0), ('promoter_pct',    45.0),
                    ('fii_pct',         15.0), ('fcf_positive_3y',  0.5),
                    ('debt_reducing',    0.5), ('screener_de',     50.0),
                ]:
                    try:
                        f[k] = float(r.get(k) or default)
                    except Exception:
                        f[k] = default
            except Exception:
                pass

            try:
                import yfinance as _yf2
                fi = _yf2.Ticker(f"{symbol}.NS").fast_info
                price = getattr(fi, 'last_price', None)
                eps_latest = f.get('eps_latest_approx', None)
                eps_cagr   = f.get('eps_cagr_5y', 8.0)
                try:
                    _sdf2    = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                           'screener_fundamentals.csv'))
                    _csv_sym2 = {'LTM': 'LTIM'}.get(symbol, symbol)
                    _row2    = _sdf2[_sdf2['symbol'] == _csv_sym2]
                    if not _row2.empty:
                        eps_latest = float(_row2.iloc[0].get('eps_latest') or 0) or None
                except Exception:
                    pass
                if price and eps_latest and float(eps_latest) > 0:
                    f['pe_ratio']  = round(float(price) / float(eps_latest), 1)
                    f['peg_ratio'] = round(f['pe_ratio'] / max(float(eps_cagr), 0.1), 2)
            except Exception:
                pass

            X    = pd.DataFrame([{k: f[k] for k in features}])
            prob = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])
            result["ml"] = {
                "ml_score":     round(prob * 100, 1),
                "prediction":   "OUTPERFORM" if pred == 1 else "UNDERPERFORM",
                "accuracy":     round(accuracy * 100, 1),
                "rsi":          round(f['rsi'], 1),
                "pos52_pct":    round(f['pos52'] * 100, 1),
                "ret_1m_pct":   round(ret_1m * 100, 1),
                "ret_3m_pct":   round(ret_3m * 100, 1),
                "golden_cross": bool(f['golden_cross']),
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[fetch_ml ERROR] {tb}", flush=True)
            result["ml"] = {"error": str(e), "traceback": tb}

    def fetch_fundamentals():
        try:
            import pandas as pd
            path = os.path.join(os.path.dirname(__file__), 'screener_fundamentals.csv')
            if not os.path.exists(path):
                result["fundamentals"] = {}
                return
            sdf = pd.read_csv(path)
            # Some symbols stored differently in CSV (e.g. VBL → VBLLTD)
            SYMBOL_CSV_MAP = {
                'LTM': 'LTIM',
            }
            csv_sym = SYMBOL_CSV_MAP.get(symbol, symbol)
            row = sdf[sdf['symbol'] == csv_sym]
            if not row.empty:
                r = row.iloc[0].to_dict()
                result["fundamentals"] = {
                    # Core metrics
                    "roce":              r.get('roce_latest_pct'),
                    "roce_latest_pct":   r.get('roce_latest_pct'),
                    "roce_avg_5y":       r.get('roce_avg_5y'),
                    "roce_trend_5y":     r.get('roce_trend_5y'),
                    # Growth
                    "sales_cagr_5y":     r.get('sales_cagr_5y'),
                    "sales_cagr_10y":    r.get('sales_cagr_10y'),
                    "sales_growth_1y":   r.get('sales_growth_1y'),
                    "profit_cagr_5y":    r.get('profit_cagr_5y'),
                    "profit_cagr_10y":   r.get('profit_cagr_10y'),
                    "profit_growth_1y":  r.get('profit_growth_1y'),
                    "eps_cagr_5y":       r.get('eps_cagr_5y'),
                    "eps_growth_1y":     r.get('eps_growth_1y'),
                    # Margins
                    "opm_latest_pct":    r.get('opm_latest_pct'),
                    "opm_avg_5y":        r.get('opm_avg_5y'),
                    "opm_trend_5y":      r.get('opm_trend_5y'),
                    # Cash flow
                    "fcf_positive_3y":   r.get('fcf_positive_3y'),
                    "ocf_positive_3y":   r.get('ocf_positive_3y'),
                    "fcf_cagr_5y":       r.get('fcf_cagr_5y'),
                    "ocf_latest_cr":     r.get('ocf_latest_cr'),
                    # Balance sheet
                    "debt_reducing":     r.get('debt_reducing'),
                    "debt_growth_1y":    r.get('debt_growth_1y'),
                    "screener_de":       r.get('screener_de'),
                    "networth_cr":       r.get('networth_cr'),
                    # Ownership
                    "promoter_pct":      r.get('promoter_pct'),
                    "fii_pct":           r.get('fii_pct'),
                    "dii_pct":           r.get('dii_pct'),
                    # Valuation
                    "eps_latest":        r.get('eps_latest'),
                    "dividend_payout_pct": r.get('dividend_payout_pct'),
                    "profit_latest_cr":  r.get('profit_latest_cr'),
                    "sales_latest_cr":   r.get('sales_latest_cr'),
                    # Scores
                    "investment_score":  r.get('investment_score'),
                    "investment_grade":  r.get('investment_grade'),
                }
            else:
                result["fundamentals"] = {}
        except Exception:
            result["fundamentals"] = {}

    def fetch_chart_insights():
        # Cache insights per symbol for 6 hours — Screener data doesn't change intraday
        _ci_cache = getattr(fetch_chart_insights, '_cache', {})
        fetch_chart_insights._cache = _ci_cache
        cached = _ci_cache.get(symbol)
        if cached and (time.time() - cached['ts']) < 21600:
            result['chart_insights'] = cached['data']
            return
        try:
            from screener_scraper import get_page, parse_table, parse_num
            import math

            SYMBOL_MAP = {'LTM': 'LTIMINDTREE', 'M&M': 'M&M', 'BAJAJ-AUTO': 'BAJAJ-AUTO'}
            scr_sym = SYMBOL_MAP.get(symbol, symbol)
            soup = get_page(scr_sym)
            pl   = parse_table(soup, 'profit-loss')
            bs   = parse_table(soup, 'balance-sheet')
            rat  = parse_table(soup, 'ratios')

            def clean(arr):
                return [float(v) for v in arr if v is not None and not math.isnan(float(v if v else 0))]

            def trend_desc(vals, label, unit='', higher_is_good=True):
                if len(vals) < 3:
                    return None
                recent  = vals[-3:]
                older   = vals[:3]
                avg_rec = sum(recent) / len(recent)
                avg_old = sum(older)  / len(older)
                if avg_old == 0:
                    return None
                chg = (avg_rec - avg_old) / abs(avg_old) * 100

                if higher_is_good:
                    if chg > 20:   trend, quality = "strong uptrend", "good"
                    elif chg > 5:  trend, quality = "gradual uptrend", "good"
                    elif chg > -5: trend, quality = "relatively flat", "neutral"
                    elif chg > -20:trend, quality = "gradual decline", "bad"
                    else:          trend, quality = "sharp decline", "bad"
                else:
                    if chg > 20:   trend, quality = "sharp increase", "bad"
                    elif chg > 5:  trend, quality = "gradual increase", "bad"
                    elif chg > -5: trend, quality = "relatively flat", "neutral"
                    elif chg > -20:trend, quality = "gradual reduction", "good"
                    else:          trend, quality = "sharp reduction", "good"

                latest = vals[-1]
                return {"trend": trend, "quality": quality, "change_pct": round(chg, 1),
                        "latest": round(latest, 1), "unit": unit}

            sales  = clean(pl.get('Sales', pl.get('Revenue', [])))
            profit = clean(pl.get('Net Profit', []))
            roce   = clean(rat.get('ROCE %', rat.get('ROE %', [])))
            debt   = clean(bs.get('Borrowings', []))
            eq_cap = clean(bs.get('Equity Capital', []))
            res    = clean(bs.get('Reserves', []))
            equity = [e + r for e, r in zip(eq_cap, res)] if eq_cap and res else []

            insights = {}
            r = trend_desc(sales,  'Revenue',  '₹ Cr', higher_is_good=True)
            if r: insights['revenue'] = {**r, 'summary': f"Revenue shows a {r['trend']} — {'positive sign of business growth' if r['quality']=='good' else 'watch for demand slowdown' if r['quality']=='bad' else 'stable but limited growth'}."}

            p = trend_desc(profit, 'Profit',   '₹ Cr', higher_is_good=True)
            if p: insights['profit'] = {**p, 'summary': f"Net profit in a {p['trend']} — {'earnings are expanding, good for shareholders' if p['quality']=='good' else 'profitability is under pressure' if p['quality']=='bad' else 'margins holding steady'}."}

            rc = trend_desc(roce,  'ROCE',     '%',    higher_is_good=True)
            if rc: insights['roce'] = {**rc, 'summary': f"ROCE is in a {rc['trend']} — {'capital is being deployed more efficiently' if rc['quality']=='good' else 'returns on capital are declining, needs monitoring' if rc['quality']=='bad' else 'capital efficiency is stable'}."}

            d = trend_desc(debt,   'Debt',     '₹ Cr', higher_is_good=False)
            if d: insights['debt'] = {**d, 'summary': f"Debt shows a {d['trend']} — {'balance sheet is strengthening' if d['quality']=='good' else 'rising debt increases financial risk' if d['quality']=='bad' else 'debt levels are stable'}."}

            eq = trend_desc(equity,'Equity',   '₹ Cr', higher_is_good=True)
            if eq: insights['equity'] = {**eq, 'summary': f"Equity in a {eq['trend']} — {'net worth is building up, good sign' if eq['quality']=='good' else 'equity base is eroding, concerning' if eq['quality']=='bad' else 'equity is stable'}."}

            result['chart_insights'] = insights
            _ci_cache[symbol] = {'data': insights, 'ts': time.time()}
        except Exception:
            result['chart_insights'] = {}

    def fetch_valuation():
        # Use nightly cache if available — avoids yfinance call entirely
        cached_val = result.get('_val_from_cache')
        if cached_val:
            result['valuation'] = cached_val
            return

        import yfinance as yf
        import concurrent.futures
        t  = yf.Ticker(f"{symbol}.NS")
        fi = t.fast_info

        price  = getattr(fi, 'last_price', None)
        shares = getattr(fi, 'shares', None)
        pe, eps, div_yield = None, None, None

        # EPS = net income / shares — from income statement
        def _get_eps():
            try:
                inc = t.income_stmt
                if inc is not None and not inc.empty:
                    ni_row = None
                    for k in inc.index:
                        if 'net income' in str(k).lower():
                            ni_row = k
                            break
                    if ni_row is not None and shares and float(shares) > 0:
                        ni = float(inc.loc[ni_row].iloc[0])
                        return round(ni / float(shares), 2)
            except Exception:
                pass
            return None

        # Div yield = trailing 12m dividends / price
        def _get_div():
            try:
                divs = t.dividends
                if divs is not None and len(divs) > 0:
                    import pandas as pd
                    cutoff = pd.Timestamp.now(tz=divs.index.tz)
                    one_yr = divs[divs.index >= (cutoff - pd.DateOffset(years=1))]
                    annual = float(one_yr.sum())
                    if annual > 0 and price and float(price) > 0:
                        return round(annual / float(price), 6)
            except Exception:
                pass
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_eps = ex.submit(_get_eps)
            f_div = ex.submit(_get_div)
            try:
                eps = f_eps.result(timeout=6)
            except Exception:
                eps = None
            try:
                div_yield = f_div.result(timeout=6)
            except Exception:
                div_yield = None

        if price and eps and float(eps) > 0:
            pe = round(float(price) / float(eps), 1)

        result["valuation"] = {
            "pe_ratio":        pe,
            "pb_ratio":        None,
            "profit_margin":   None,
            "revenue_growth":  None,
            "earnings_growth": None,
            "debt_to_equity":  None,
            "dividend_yield":  div_yield,
            "eps":             eps,
        }

    def fetch_sentiment():
        try:
            from news_sentiment import get_sentiment_score
            from ml_screener import _cache
            # Try disk cache first (6h TTL)
            disk_sent = load_disk_cache(f'sent_{symbol}', max_age_hours=6)
            if disk_sent:
                result["sentiment"] = disk_sent
                return
            sent = get_sentiment_score(symbol)

            sent['fetched_at'] = datetime.now().isoformat()

            cached_sent = _cache.get(f'sent_{symbol}', {})
            if cached_sent.get('data') and cached_sent.get('ts'):
                age_hours = (time.time() - cached_sent['ts']) / 3600
                if age_hours > 24:
                    raw = float(cached_sent['data'].get('sentiment_score', 0))
                    decay = max(0.1, 1 - (age_hours - 24) / 72)
                    sent['sentiment_score'] = round(raw * decay, 1)
                    sent['sentiment_label'] = (
                        'positive' if sent['sentiment_score'] > 8  else
                        'negative' if sent['sentiment_score'] < -8 else
                        'neutral'
                    )
                    sent['dampened'] = True

            _cache[f'sent_{symbol}'] = {'data': sent, 'ts': time.time()}
            save_disk_cache(f'sent_{symbol}', sent)
            result["sentiment"] = sent
        except Exception:
            result["sentiment"] = {"sentiment_score": 0, "sentiment_label": "neutral"}

    def fetch_macro():
        try:
            from ml_screener import _cache
            from macro_sentiment import get_macro_sentiment, apply_macro_to_stock as _apply
            macro_cache = _cache.get('macro_news', {})
            # Use cache if it's warm (filled by ml-screen endpoint), else fetch fresh
            if macro_cache.get('data') and macro_cache.get('ts', 0) > 0:
                macro_data = macro_cache['data']
            else:
                # Fetch fresh (takes ~30s for all 26 topics, so do it in background)
                macro_data = get_macro_sentiment()
                _cache['macro_news']['data'] = macro_data
                _cache['macro_news']['ts']   = time.time()
                save_disk_cache('macro', macro_data)
            result["macro"] = _apply(symbol, macro_data)
        except Exception:
            result["macro"] = {"macro_score": 0, "macro_label": "neutral"}

    # ── Launch all threads simultaneously ─────────────────────────────
    threads = [
        threading.Thread(target=fetch_quote),
        threading.Thread(target=fetch_ml),
        threading.Thread(target=fetch_fundamentals),
        threading.Thread(target=fetch_valuation),
        threading.Thread(target=fetch_sentiment),
        threading.Thread(target=fetch_macro),
        threading.Thread(target=fetch_chart_insights),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=40)

    # ── Combined score ────────────────────────────────────────────────
    try:
        ml_raw  = result.get("ml", {}).get("ml_score", 50) or 50

        # ══════════════════════════════════════════════════════════════
        # CUSTOM FUNDAMENTAL SCORE — replaces black-box Screener score
        # Built from raw CSV fields across 4 categories:
        #   30% Growth Quality
        #   25% Profitability & Capital Efficiency
        #   25% Financial Health
        #   20% Management Quality
        # ══════════════════════════════════════════════════════════════
        _fund_raw  = result.get("fundamentals", {})
        _quote_raw = result.get("quote", {})
        _sec_str   = str(_quote_raw.get("industry", "") or "").lower()

        _is_bank    = any(x in _sec_str for x in ['bank','nbfc','financ','insurance','microfinance'])
        _is_it      = any(x in _sec_str for x in ['it','software','technolog','computer'])
        _is_fmcg    = any(x in _sec_str for x in ['fmcg','consumer','food','beverag'])
        _is_pharma  = any(x in _sec_str for x in ['pharma','health','medical','hospital'])
        _is_defence = any(x in _sec_str for x in ['defence','shipbuild','aerospace'])
        _is_metal   = any(x in _sec_str for x in ['metal','steel','alumin','mining'])
        _is_infra   = any(x in _sec_str for x in ['infra','construct','cement','road','power'])

        def _f(key, default=0.0):
            v = _fund_raw.get(key)
            try: return float(v) if v not in (None,'','None','nan') else default
            except: return default

        _sales_cagr5  = _f('sales_cagr_5y')
        _sales_cagr10 = _f('sales_cagr_10y')
        _sales_1y     = _f('sales_growth_1y')
        _prof_cagr5   = _f('profit_cagr_5y')
        _prof_cagr10  = _f('profit_cagr_10y')
        _prof_1y      = _f('profit_growth_1y')
        _eps_cagr5    = _f('eps_cagr_5y')
        _opm_lat      = _f('opm_latest_pct')
        _opm_avg      = _f('opm_avg_5y')
        _opm_trend    = _f('opm_trend_5y')
        _roce_lat     = _f('roce_latest_pct')
        _roce_avg     = _f('roce_avg_5y', _roce_lat)
        _roce_trend   = _f('roce_trend_5y')
        _de           = _f('screener_de', None)
        _debt_gr      = _f('debt_growth_1y')
        _debt_red     = str(_fund_raw.get('debt_reducing','')).lower() == 'true'
        _fcf_ok       = str(_fund_raw.get('fcf_positive_3y','')).lower() == 'true'
        _ocf_ok       = str(_fund_raw.get('ocf_positive_3y','')).lower() == 'true'
        _fcf_cagr     = _f('fcf_cagr_5y')
        _promoter     = _f('promoter_pct')
        _fii          = _f('fii_pct')
        _dii          = _f('dii_pct')
        _div_payout   = _f('dividend_payout_pct')

        # ── Clamp extreme outliers ────────────────────────────────────
        _opm_lat  = max(_opm_lat, -100.0)
        _opm_avg  = max(_opm_avg, -100.0)

        # ── Category 1: Growth Quality (0–100) ────────────────────────
        g = 50

        if _sales_cagr5 > 25:    g += 18
        elif _sales_cagr5 > 18:  g += 12
        elif _sales_cagr5 > 12:  g += 6
        elif _sales_cagr5 > 5:   g += 0
        elif _sales_cagr5 < 0:   g -= 15
        else:                    g -= 5

        if _sales_cagr10 > 0 and _sales_cagr5 > _sales_cagr10 * 1.2:
            g += 8
        elif _sales_cagr5 > 0 and _sales_cagr10 > 0 and _sales_cagr5 < _sales_cagr10 * 0.6:
            g -= 8

        _prof_weight = 0.6 if _is_metal else 1.0
        if _prof_cagr5 > 25:     g += int(15 * _prof_weight)
        elif _prof_cagr5 > 15:   g += int(8  * _prof_weight)
        elif _prof_cagr5 > 8:    g += int(3  * _prof_weight)
        elif _prof_cagr5 < 0:    g -= 12
        elif _prof_cagr5 < 5:    g -= 5

        if _sales_1y > 20 and _prof_1y > 15:   g += 5
        elif _sales_1y < -5 or _prof_1y < -10: g -= 8

        g = max(0, min(100, g))

        # ── Category 2: Profitability & Capital Efficiency (0–100) ────
        p = 50

        if not _is_bank:
            if _roce_lat > 35:    p += 22
            elif _roce_lat > 25:  p += 15
            elif _roce_lat > 18:  p += 8
            elif _roce_lat > 12:  p += 2
            elif _roce_lat < 8:   p -= 12
            elif _roce_lat < 0:   p -= 20

            if _roce_lat > _roce_avg + 5:    p += 10
            elif _roce_lat > _roce_avg + 2:  p += 5
            elif _roce_lat < _roce_avg - 5:  p -= 10
            elif _roce_lat < _roce_avg - 2:  p -= 5
        else:
            if _prof_cagr5 > 20:   p += 15
            elif _prof_cagr5 > 12: p += 8
            elif _prof_cagr5 < 5:  p -= 8

        if not _is_bank:
            if _opm_lat > 30:     p += 12
            elif _opm_lat > 20:   p += 7
            elif _opm_lat > 12:   p += 2
            elif _opm_lat < 0:    p -= 15
            elif _opm_lat < 5:    p -= 8

            if _opm_trend > 5:    p += 6
            elif _opm_trend < -5: p -= 6

        p = max(0, min(100, p))

        # ── Category 3: Financial Health (0–100) ──────────────────────
        h = 50

        if _fcf_ok:               h += 15
        elif _ocf_ok:             h += 7
        else:
            if not _is_bank:  h -= 12

        if _fcf_cagr > 20:        h += 8
        elif _fcf_cagr > 10:      h += 4
        elif _fcf_cagr < -10:     h -= 6

        if not _is_bank and _de is not None:
            if _de < 0.1:         h += 12
            elif _de < 0.3:       h += 8
            elif _de < 0.6:       h += 3
            elif _de < 1.0:       h -= 3
            elif _de < 2.0:
                if _sales_1y < 15: h -= 10
                else:              h -= 4
            else:
                h -= 18

            if _debt_red:         h += 8
            elif _debt_gr > 30 and _sales_1y < 10:
                h -= 8

        h = max(0, min(100, h))

        # ── Category 4: Management Quality (0–100) ────────────────────
        m = 50

        _is_mnc = _promoter == 0 and _fii > 25
        if _is_mnc:
            m += 5
        else:
            if _promoter > 65:    m += 15
            elif _promoter > 55:  m += 10
            elif _promoter > 45:  m += 4
            elif _promoter > 25:  m += 0
            elif _promoter < 15:  m -= 8

        _inst_total = _fii + _dii
        if _inst_total > 50:      m += 10
        elif _inst_total > 35:    m += 6
        elif _inst_total > 20:    m += 2
        elif _inst_total < 10:    m -= 5

        if 20 <= _div_payout <= 60:   m += 6
        elif _div_payout > 100:        m -= 8
        elif _div_payout == 0:         m -= 2

        m = max(0, min(100, m))

        # ── Weighted final score ───────────────────────────────────────
        scr_raw = round(
            g * 0.30 +   # Growth Quality
            p * 0.25 +   # Profitability & Capital Efficiency
            h * 0.25 +   # Financial Health
            m * 0.20,    # Management Quality
            1)

        pe = result.get("valuation", {}).get("pe_ratio") or 20

        # ── PEG ratio — PE relative to growth ────────────────────────
        # Use minimum of 5Y CAGR and recent 1Y growth to avoid peak-cycle inflation
        _recent_growth = min(_eps_cagr5, _prof_1y) if _prof_1y > 0 else _eps_cagr5
        growth_for_peg = max(_recent_growth, 1)
        peg = round(pe / growth_for_peg, 2) if growth_for_peg > 0 else None

        yfin_score = 50

        # ── 1. PE scoring — contextual by sector ─────────────────────
        if _is_bank:
            sector_pe_fair = 14
        elif _is_fmcg or _is_it:
            sector_pe_fair = 32
        elif _is_pharma or _is_defence:
            sector_pe_fair = 28
        elif _is_metal or _is_infra:
            sector_pe_fair = 14
        else:
            sector_pe_fair = 22

        pe_vs_sector = pe / sector_pe_fair if sector_pe_fair > 0 else 1.0
        if pe_vs_sector < 0.7:    yfin_score += 12
        elif pe_vs_sector < 0.9:  yfin_score += 7
        elif pe_vs_sector < 1.1:  yfin_score += 3
        elif pe_vs_sector < 1.4:  yfin_score -= 3
        else:                     yfin_score -= 8

        # ── 2. PEG override — growth justifies PE ─────────────────────
        # PEG < 1 means earnings growing faster than PE — genuinely cheap
        # PEG > 3 means paying a lot more than growth warrants — expensive
        if peg is not None:
            if peg < 0.5:    yfin_score += 15
            elif peg < 0.8:  yfin_score += 10
            elif peg < 1.2:  yfin_score += 5
            elif peg < 2.0:  yfin_score -= 2
            elif peg < 3.0:  yfin_score -= 6
            else:            yfin_score -= 12

        # ── 3. Margin scoring — contextual ───────────────────────────
        if not _is_bank:
            if _opm_lat > 25:     yfin_score += 10
            elif _opm_lat > 15:   yfin_score += 5
            elif _opm_lat < 0:    yfin_score -= 15
            elif _opm_lat < 5:    yfin_score -= 8
            if _opm_trend > 3:    yfin_score += 5
            elif _opm_trend < -3: yfin_score -= 5

        # ── 4. Growth scoring ─────────────────────────────────────────
        if _sales_cagr5 > 20:     yfin_score += 10
        elif _sales_cagr5 > 12:   yfin_score += 5
        elif _sales_cagr5 < 0:    yfin_score -= 10
        elif _sales_cagr5 < 5:    yfin_score -= 4

        if _prof_cagr5 > 20:      yfin_score += 8
        elif _prof_cagr5 > 12:    yfin_score += 4
        elif _prof_cagr5 < 0:     yfin_score -= 8

        # ── 5. ROCE scoring ───────────────────────────────────────────
        if not _is_bank:
            if _roce_lat > 25:    yfin_score += 10
            elif _roce_lat > 15:  yfin_score += 5
            elif _roce_lat < 8:   yfin_score -= 10
            if _roce_lat > _roce_avg + 3:   yfin_score += 5
            elif _roce_lat < _roce_avg - 3: yfin_score -= 5

        # ── 6. Debt scoring — contextual ─────────────────────────────
        if not _is_bank:
            if _debt_red:
                yfin_score += 6
            elif _debt_gr > 20:
                if _sales_1y > 15 and _prof_1y > 10:
                    yfin_score += 0
                elif _sales_1y > 0:
                    yfin_score -= 4
                else:
                    yfin_score -= 12
            elif _debt_gr > 0:
                yfin_score -= 2

        # ── 7. FCF / Cash flow quality ────────────────────────────────
        if _fcf_ok:               yfin_score += 8
        if _ocf_ok:               yfin_score += 3

        # ── 8. Promoter confidence ────────────────────────────────────
        if _promoter > 60:        yfin_score += 5
        elif _promoter > 50:      yfin_score += 3
        elif _promoter < 25:      yfin_score -= 5

        yfin_score = max(0, min(100, yfin_score))

        sent_raw   = result.get("sentiment", {}).get("sentiment_score", 0) or 0
        sent_score = max(0, min(100, 50 + sent_raw * 0.5))

        macro_raw   = result.get("macro", {}).get("macro_score", 0) or 0
        macro_score = max(0, min(100, 50 + macro_raw * 0.5))

        # ── Sentiment gated by magnitude ──────────────────────────────
        # Small daily noise (|score| < 15) → excluded from combined
        # Significant news (15-40) → small weight
        # Crisis-level news (>40) → hard override applied after
        sent_impact  = 0
        macro_impact = 0
        if abs(sent_raw) >= 15:
            sent_impact  = max(0, min(100, 50 + sent_raw * 0.5))
        if abs(macro_raw) >= 15:
            macro_impact = max(0, min(100, 50 + macro_raw * 0.5))

        if sent_impact or macro_impact:
            combined = round(
                ml_raw       * 0.22 +
                scr_raw      * 0.41 +
                yfin_score   * 0.25 +
                sent_impact  * 0.07 +
                macro_impact * 0.05, 1)
            if ml_raw > 65 and scr_raw > 75:
                combined = round(combined - 2.0, 1)
        else:
            # No significant sentiment — stable score
            combined = round(
                ml_raw     * 0.25 +
                scr_raw    * 0.45 +
                yfin_score * 0.30, 1)
            # Correlation penalty — ML and fundamentals share inputs
            if ml_raw > 65 and scr_raw > 75:
                combined = round(combined - 2.0, 1)

        combined = max(0, min(100, combined))

        # ── Crisis override — extreme negative sentiment ───────────────
        if sent_raw < -40 and scr_raw < 70:  # don't override strong businesses
            combined      = min(combined, 35)
            verdict       = 'SELL'
            verdict_color = 'red'
        elif sent_raw < -25 and combined >= 50:
            combined      = min(combined, 49)

        grade = 'A+' if combined >= 82 else 'A' if combined >= 68 else 'B' if combined >= 58 else 'C' if combined >= 48 else 'D'

        # ── Base verdict with conviction tiers ────────────────────────
        if combined >= 82:   verdict, verdict_color = 'STRONG BUY', 'green'
        elif combined >= 68: verdict, verdict_color = 'BUY',        'green'
        elif combined >= 58: verdict, verdict_color = 'MILD BUY',   'green'
        elif combined >= 48: verdict, verdict_color = 'HOLD',       'gold'
        elif combined >= 38: verdict, verdict_color = 'MILD SELL',  'red'
        else:                verdict, verdict_color = 'SELL',       'red'

        # ── Short-term verdict (ML + technicals) ─────────────────────
        ml_s         = float(ml_raw or 50)
        short_verdict = 'BUY' if ml_s >= 62 else 'SELL' if ml_s < 40 else 'HOLD'

        # ── Long-term verdict — deferred until val_signal is computed below ──
        fund_score_v = float(scr_raw or 50)
        long_verdict = 'HOLD'  # will be refined after val_signal is computed

        # ── Contrarian override ───────────────────────────────────────
        rsi_val  = float(result.get('ml', {}).get('rsi') or 50)
        pos52    = float(result.get('ml', {}).get('pos52_pct') or 50)
        ret_1m   = float(result.get('ml', {}).get('ret_1m_pct') or 0)

        contrarian = (
            sent_raw  < -25  and
            scr_raw   >= 60  and
            rsi_val   < 45   and
            pos52     < 40   and
            ret_1m    < -5   and
            combined  >= 42
        )

        strong_contrarian = (
            contrarian       and
            scr_raw   >= 75  and
            rsi_val   < 35   and
            pos52     < 25
        )

        if strong_contrarian:
            verdict       = 'STRONG BUY'
            verdict_color = 'green'
        elif contrarian:
            verdict       = 'BUY'
            verdict_color = 'green'

        # ── Risk level ────────────────────────────────────────────────
        mcap_str = result.get('quote', {}).get('market_cap', '') or ''
        try:
            mc_val   = float(mcap_str.replace('₹','').replace('L Cr','')
                             .replace('T Cr','').replace('Cr','')
                             .replace(',','').strip() or 0)
            is_large = 'L Cr' in mcap_str and mc_val > 50
        except Exception:
            is_large = False

        vol      = abs(float(result.get('ml', {}).get('ret_1m_pct') or 0))
        rsi_risk = float(result.get('ml', {}).get('rsi') or 50)
        pos52_r  = float(result.get('ml', {}).get('pos52_pct') or 50)

        risk_points = 0

        # ── Size ──────────────────────────────────────────────────────
        if is_large:           risk_points += 0
        else:                  risk_points += 2

        # ── Debt — contextual by sector ───────────────────────────────
        if _is_bank:
            risk_points += 0
        elif _debt_red:
            risk_points += 0
        elif _de is None or _de < 0.3:   risk_points += 0
        elif _de < 0.8:
            if _sales_1y > 15 and _prof_1y > 10:
                risk_points += 0
            else:
                risk_points += 1
        elif _de < 1.5:
            if _sales_1y > 20:  risk_points += 1
            else:               risk_points += 2
        else:
            risk_points += 3

        # ── Margin stability ──────────────────────────────────────────
        if not _is_bank:
            if _opm_trend < -5:   risk_points += 2
            elif _opm_trend < -2: risk_points += 1
            if _opm_lat < 0:      risk_points += 2

        # ── FCF health ────────────────────────────────────────────────
        if not _fcf_ok:          risk_points += 1

        # ── Volatility ────────────────────────────────────────────────
        if vol < 5:              risk_points += 0
        elif vol < 12:           risk_points += 1
        else:                    risk_points += 2

        # ── RSI extremes ──────────────────────────────────────────────
        if rsi_risk > 78:        risk_points += 1
        elif rsi_risk < 28:      risk_points += 1

        # ── 52W position ──────────────────────────────────────────────
        if pos52_r < 15:         risk_points += 1

        # ── Fundamental quality ───────────────────────────────────────
        if scr_raw < 45:         risk_points += 2
        elif scr_raw >= 70:      risk_points -= 1

        # ── Growth trajectory ─────────────────────────────────────────
        if _sales_cagr5 < 0 and _prof_cagr5 < 0:
            risk_points += 2
        elif _sales_cagr5 > 15 and _prof_cagr5 > 15:
            risk_points -= 1

        risk_points = max(0, risk_points)
        if risk_points <= 2:   risk, risk_color = 'Low',    'green'
        elif risk_points <= 5: risk, risk_color = 'Medium', 'gold'
        else:                  risk, risk_color = 'High',   'red'

        # ── Reason ────────────────────────────────────────────────────
        reasons = []
        if strong_contrarian:
            reasons.append('Heavily oversold — strong buy opportunity')
        elif contrarian:
            reasons.append('Oversold on strong fundamentals')
        else:
            if scr_raw >= 70:    reasons.append('Strong fundamentals')
            elif scr_raw < 45:   reasons.append('Weak fundamentals')
            if ml_raw >= 60:     reasons.append('positive ML signal')
            elif ml_raw < 40:    reasons.append('negative ML signal')
            if sent_raw > 10:    reasons.append('positive news')
            elif sent_raw < -10: reasons.append('negative news')
            if macro_raw > 10:   reasons.append('favourable macro')
            elif macro_raw < -10:reasons.append('unfavourable macro')
            if not reasons:      reasons.append('Mixed signals')

        reason = reasons[0].capitalize()
        if len(reasons) > 1:
            reason += ', ' + ', '.join(reasons[1:3])
        score_10 = round(combined / 10, 1)

        # ── Valuation Signal + Buy Zone ───────────────────────────────
        val_signal = None
        try:
            import pandas as _pd
            _sdf     = _pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'screener_fundamentals.csv'))
            _csv_sym = {'LTM': 'LTIM'}.get(symbol, symbol)
            _row     = _sdf[_sdf['symbol'] == _csv_sym]
            _r       = _row.iloc[0].to_dict() if not _row.empty else {}

            eps_from_valuation = result.get("valuation", {}).get("eps")
            eps_latest_raw = float(eps_from_valuation or _r.get('eps_latest') or 0)
            eps_cagr_raw   = float(_r.get('eps_cagr_5y') or 8)
            _opm_lat_v     = float(_r.get('opm_latest_pct') or 0)
            _opm_avg_v     = float(_r.get('opm_avg_5y')     or _opm_lat_v)
            _prof_cagr_10  = float(_r.get('profit_cagr_10y') or eps_cagr_raw)

            # ── Cyclicality detection ──────────────────────────────────
            _sec_v     = str(result.get('quote', {}).get('industry', '') or '').lower()
            _is_cyc_v  = any(x in _sec_v for x in [
                'metal', 'steel', 'alumin', 'mining', 'oil', 'petro', 'refin',
                'fertiliser', 'chemical', 'coal', 'gas', 'cement'
            ])

            # ── Normalise EPS for cyclicals ────────────────────────────
            if _is_cyc_v and _opm_lat_v > 0 and _opm_avg_v > 0:
                margin_ratio = _opm_avg_v / _opm_lat_v
                margin_ratio = min(margin_ratio, 1.0)
                eps_latest   = round(eps_latest_raw * margin_ratio, 2)
                eps_cagr     = min(eps_cagr_raw, _prof_cagr_10)
                eps_cagr     = min(eps_cagr, 15.0)
            else:
                eps_latest = eps_latest_raw
                eps_cagr   = eps_cagr_raw
            roce_l      = float(_r.get('roce_latest_pct') or 10)
            roce_a      = float(_r.get('roce_avg_5y')  or roce_l)
            fcf_ok      = bool(_r.get('fcf_positive_3y'))
            debt_red    = bool(_r.get('debt_reducing'))
            cur_pe      = float(result.get('valuation', {}).get('pe_ratio') or 0)
            cur_price   = float(str(result.get('quote', {}).get('price') or 0)
                                .replace(',', '')) or None

            if cur_price:
                if eps_latest > 0:
                    # ── Normal profitable company ─────────────────────────
                    rbi_rate = RBI_REPO_RATE
                    rate_adj = 4.4 / rbi_rate

                    SECTOR_PE = {
                        'IT': 28, 'Technology': 28, 'Software': 28,
                        'FMCG': 35, 'Consumer': 32,
                        'Pharma': 30, 'Healthcare': 30,
                        'Banking': 15, 'Finance': 18, 'NBFC': 18,
                        'Auto': 20, 'Automobile': 20,
                        'Metals': 10, 'Steel': 10, 'Mining': 10,
                        'Energy': 12, 'Oil': 12, 'Power': 14,
                        'Infrastructure': 18, 'Construction': 16,
                        'Cement': 20, 'Real Estate': 20,
                        'Defence': 30, 'Chemicals': 22,
                    }
                    sector_str = str(result.get('quote', {}).get('industry', '') or '')
                    base_pe = 22
                    for k, v in SECTOR_PE.items():
                        if k.lower() in sector_str.lower():
                            base_pe = v
                            break
                # Explicit bank override — banks always use lower PE
                if any(x in sector_str.lower() for x in ['bank','nbfc','financ','insurance','microfinance']):
                    base_pe = min(base_pe, 15)

                    quality_mult = 1.0
                    roce_l2  = float(_r.get('roce_latest_pct') or 10)
                    roce_a2  = float(_r.get('roce_avg_5y') or roce_l2)
                    if roce_l2 > roce_a2 + 3:   quality_mult += 0.10
                    if bool(_r.get('fcf_positive_3y')): quality_mult += 0.08
                    if bool(_r.get('debt_reducing')):    quality_mult += 0.07
                    if float(_r.get('promoter_pct') or 0) > 55: quality_mult += 0.05
                    quality_mult = min(quality_mult, 1.35)

                    opm_trend = float(_r.get('opm_trend_5y') or 0)
                    reliable_growth = min(float(eps_cagr or 8.0), 25)
                    if opm_trend < -3:
                        reliable_growth = min(reliable_growth, 15)

                    _fair_pe_cap = 70 if any(x in sector_str.lower() for x in
                                   ['it','software','technolog','defence','shipbuild',
                                    'pharma','health','fmcg','consumer']) else 55
                    fair_pe = min(
                        (base_pe + 1.5 * reliable_growth) * rate_adj * quality_mult,
                        _fair_pe_cap
                    )
                    fair_pe    = round(fair_pe, 1)
                    fair_value = round(eps_latest * fair_pe, 1)

                    if scr_raw >= 75:   mos = 0.10
                    elif scr_raw >= 60: mos = 0.15
                    else:               mos = 0.25
                    if _is_bank:   mos += 0.05
                    if _is_cyc_v:  mos += 0.05
                    if not fcf_ok: mos += 0.05
                    mos = min(mos, 0.35)

                    buy_zone_high = round(fair_value * (1 - mos * 0.5), 1)
                    buy_zone_low  = round(fair_value * (1 - mos), 1)
                    pct_vs_fair   = round((cur_price - fair_value) / fair_value * 100, 1) \
                                    if fair_value > 0 else 0

                    confidence = 50
                    if eps_cagr > 15:    confidence += 15
                    elif eps_cagr > 10:  confidence += 10
                    elif eps_cagr > 5:   confidence += 5
                    elif eps_cagr < 0:   confidence -= 15
                    if roce_l > roce_a:  confidence += 10
                    if fcf_ok:           confidence += 10
                    if debt_red:         confidence += 5
                    if scr_raw >= 75:    confidence += 10
                    elif scr_raw < 45:   confidence -= 10
                    if cur_pe <= 0:      confidence -= 10
                    confidence = max(20, min(90, confidence))

                    is_quality   = scr_raw >= 60
                    is_expensive = cur_price > fair_value * 1.15
                    is_cheap     = cur_price < fair_value * 0.90

                    if is_quality and is_cheap:
                        sig_label = "Undervalued Quality"
                        sig_color = "green"
                        sig_desc  = "Strong business trading below fair value — opportunity"
                    elif is_quality and is_expensive:
                        sig_label = "Overvalued Quality"
                        sig_color = "gold"
                        sig_desc  = "Strong business but priced above fair value — wait for dip"
                    elif is_quality:
                        sig_label = "Fairly Valued Quality"
                        sig_color = "green"
                        sig_desc  = "Strong business at a fair price"
                    elif is_cheap:
                        sig_label = "Value Trap Risk"
                        sig_color = "red"
                        sig_desc  = "Cheap valuation but weak fundamentals — be cautious"
                    else:
                        sig_label = "Overpriced Weak Business"
                        sig_color = "red"
                        sig_desc  = "Weak fundamentals and expensive — avoid"

                    val_signal = {
                        "label":         sig_label,
                        "color":         sig_color,
                        "description":   sig_desc,
                        "fair_value":    fair_value,
                        "fair_pe":       round(fair_pe, 1),
                        "current_pe":    cur_pe if cur_pe > 0 else None,
                        "pct_vs_fair":   pct_vs_fair,
                        "buy_zone_low":  buy_zone_low,
                        "buy_zone_high": buy_zone_high,
                        "confidence":    confidence,
                        "current_price": cur_price,
                    }

                else:
                    # ── Loss-making company ───────────────────────────────
                    _sales_cagr  = float(_r.get('sales_cagr_5y')    or 0)
                    _prof_gr_1y  = float(_r.get('profit_growth_1y') or 0)
                    _prof_cagr5  = float(_r.get('profit_cagr_5y')   or -999)
                    _roce_loss   = float(_r.get('roce_latest_pct')  or 0)

                    _is_turnaround = _prof_gr_1y > 30 and _sales_cagr > 5
                    _is_pre_profit = _sales_cagr > 20 and _roce_loss > -20
                    _is_distressed = _sales_cagr < 0 or (_prof_cagr5 < -15 and _prof_cagr5 != -999)

                    if _is_turnaround:
                        sig_label = "Turnaround In Progress"
                        sig_color = "gold"
                        sig_desc  = "Currently loss-making but profits improving rapidly — watch for EPS turning positive"
                        confidence = 35
                    elif _is_pre_profit:
                        sig_label = "Pre-Profit Growth"
                        sig_color = "gold"
                        sig_desc  = "Loss-making but revenue growing strongly — valuation based on future earnings potential"
                        confidence = 30
                    elif _is_distressed:
                        sig_label = "Distressed Business"
                        sig_color = "red"
                        sig_desc  = "Loss-making with declining revenue — high risk, avoid unless deep turnaround thesis"
                        confidence = 20
                    else:
                        sig_label = "Loss-Making"
                        sig_color = "red"
                        sig_desc  = "Company is currently not profitable — Graham valuation not applicable"
                        confidence = 25

                    _sales_cr      = float(_r.get('sales_latest_cr') or 0)
                    _ps_ratio      = None
                    _fair_value_ps = None
                    try:
                        # Use nightly cache for market cap — avoids yfinance timeout on Render
                        _nc_stock  = (get_nightly_cache() or {}).get('stocks', {}).get(symbol, {})
                        _mcap_str  = _nc_stock.get('quote', {}).get('market_cap', '') or ''
                        _mcap      = 0
                        try:
                            if 'L Cr' in _mcap_str:
                                _mcap = float(_mcap_str.replace('₹','').replace('L Cr','').strip()) * 1e12
                            elif 'T Cr' in _mcap_str:
                                _mcap = float(_mcap_str.replace('₹','').replace('T Cr','').strip()) * 1e14
                            elif 'Cr' in _mcap_str:
                                _mcap = float(_mcap_str.replace('₹','').replace('Cr','').strip()) * 1e7
                        except Exception:
                            pass
                        # Fallback: estimate from live quote price × shares proxy
                        if _mcap == 0:
                            _cp_now    = float(result.get('quote', {}).get('price') or 0)
                            _profit_cr = float(_r.get('profit_latest_cr') or 0)
                            _eps_csv   = float(_r.get('eps_latest') or 0)
                            if _cp_now > 0 and abs(_eps_csv) > 0 and _profit_cr != 0:
                                _shares = abs(_profit_cr * 1e7 / _eps_csv)
                                _mcap   = _cp_now * _shares
                        if _mcap > 0 and _sales_cr > 0:
                            _sales_inr    = _sales_cr * 1e7
                            _ps_ratio     = round(_mcap / _sales_inr, 1)
                            _fair_ps      = 3.0 if _sales_cagr > 25 else 2.0 if _sales_cagr > 15 else 1.0
                            _eps_abs      = abs(float(_r.get('eps_latest') or 1))
                            _profit_cr2   = abs(float(_r.get('profit_latest_cr') or 0))
                            _shares_est   = (_profit_cr2 * 1e7 / _eps_abs) if _eps_abs > 0 and _profit_cr2 > 0 else 1
                            _fair_value_ps = round((_sales_inr * _fair_ps) / _shares_est, 1) if _shares_est > 0 else None
                    except Exception:
                        pass

                    val_signal = {
                        "label":         sig_label,
                        "color":         sig_color,
                        "description":   sig_desc,
                        "fair_value":    _fair_value_ps,
                        "fair_pe":       None,
                        "current_pe":    None,
                        "pct_vs_fair":   round((cur_price - _fair_value_ps) / _fair_value_ps * 100, 1)
                                         if _fair_value_ps and _fair_value_ps > 0 else None,
                        "buy_zone_low":  round(_fair_value_ps * 0.75, 1) if _fair_value_ps else None,
                        "buy_zone_high": round(_fair_value_ps * 0.90, 1) if _fair_value_ps else None,
                        "confidence":    confidence,
                        "current_price": cur_price,
                        "ps_ratio":      _ps_ratio,
                        "pre_profit":    True,
                    }
        except Exception:
            val_signal = None

        # Compute long_verdict now that val_signal is available
        val_discount = float(val_signal.get('pct_vs_fair') or 0) if val_signal else 0.0
        lt_score     = fund_score_v * 0.5 + max(0, -val_discount) * 2.0
        long_verdict = 'BUY'  if (fund_score_v >= 60 and val_discount < -10) else \
                       'BUY'  if (lt_score >= 50 and val_discount <= 15) else \
                       'SELL' if (fund_score_v < 40 or val_discount > 40) else 'HOLD'

        # Promote using sub-verdicts — respect conviction level
        if verdict == 'HOLD' and short_verdict == 'BUY' and long_verdict == 'BUY':
            verdict, verdict_color = 'MILD BUY', 'green'
        elif verdict in ('MILD BUY', 'BUY') and short_verdict == 'BUY' and long_verdict == 'BUY' and combined >= 68:
            verdict, verdict_color = 'BUY', 'green'

        result["combined"] = {
            "score":               combined,
            "grade":               grade,
            "yfin_score":          round(yfin_score, 1),
            "sent_score":          round(sent_score, 1),
            "macro_score":         round(macro_score, 1),
            "screener_score":      round(scr_raw, 1),
            "verdict":             verdict,
            "verdict_color":       verdict_color,
            "risk":                risk,
            "risk_color":          risk_color,
            "reason":              reason,
            "score_10":            score_10,
            "valuation_signal":    val_signal,
            "short_term_verdict":  short_verdict,
            "long_term_verdict":   long_verdict,
        }
    except Exception:
        result["combined"] = {"score": 50, "grade": "C"}

    # ── Forecast ──────────────────────────────────────────────────────
    try:
        # Safe fallbacks for variables from combined score block
        _opm_trend  = _opm_trend  if '_opm_trend'  in locals() else 0.0
        _is_metal   = _is_metal   if '_is_metal'   in locals() else False
        _is_infra   = _is_infra   if '_is_infra'   in locals() else False
        _debt_gr    = _debt_gr    if '_debt_gr'    in locals() else 0.0
        _sales_1y   = _sales_1y   if '_sales_1y'   in locals() else 0.0
        _prof_1y    = _prof_1y    if '_prof_1y'    in locals() else 0.0
        scr_raw     = scr_raw     if 'scr_raw'     in locals() else 50.0

        fund  = result.get("fundamentals") or {}
        val   = result.get("valuation") or {}
        quote = result.get("quote") or {}
        ml    = result.get("ml") or {}
        sent  = result.get("sentiment") or {}
        macro = result.get("macro") or {}

        eps         = val.get("eps")
        pe          = float(val.get("pe_ratio") or 20)
        price_now   = float(str(quote.get("price") or 0).replace(",","")) or None

        # ── Base CAGRs from historical Screener data ───────────────────
        sales_cagr_5  = float(fund.get("sales_cagr_5y")   or 8)
        profit_cagr_5 = float(fund.get("profit_cagr_5y")  or 8)
        profit_cagr_10= float(fund.get("profit_cagr_10y") or profit_cagr_5)
        eps_cagr_5    = float(fund.get("eps_cagr_5y")     or profit_cagr_5)

        # ── Cyclicality detection ──────────────────────────────────────
        _fcast_sec   = str(quote.get("industry", "") or "").lower()
        _is_cyclical = any(x in _fcast_sec for x in [
            'metal', 'steel', 'alumin', 'mining', 'oil', 'petro', 'refin',
            'fertiliser', 'chemical', 'coal', 'gas', 'cement'
        ])

        if _is_cyclical:
            sales_cagr  = min(sales_cagr_5,  float(fund.get("sales_cagr_10y")  or sales_cagr_5),  12)
            profit_cagr = min(profit_cagr_5, profit_cagr_10,                                      12)
            eps_cagr    = min(eps_cagr_5,    profit_cagr_10,                                      12)
            _opm_l = float(fund.get("opm_latest_pct") or 0)
            _opm_a = float(fund.get("opm_avg_5y")     or _opm_l)
            if _opm_l > 0 and _opm_a > 0 and eps:
                _margin_adj = min(_opm_a / _opm_l, 1.0)
                eps = round(float(eps) * _margin_adj, 2)
        else:
            sales_cagr  = min(sales_cagr_5,  25)
            profit_cagr = min(profit_cagr_5, 30)
            eps_cagr    = min(eps_cagr_5,    30)
        roce_latest = float(fund.get("roce") or fund.get("roce_latest_pct") or 10)
        roce_avg    = float(fund.get("roce_avg_5y") or roce_latest)
        ocf         = fund.get("ocf_latest_cr")
        fcf_ok      = str(fund.get("fcf_positive_3y", "")).lower() == 'true'
        debt_red    = str(fund.get("debt_reducing",    "")).lower() == 'true'
        promoter    = float(fund.get("promoter_pct") or 40)

        # ── RSI & momentum from ML ─────────────────────────────────────
        rsi         = float(ml.get("rsi") or 50)
        ret_3m      = float(ml.get("ret_3m_pct") or 0)
        golden      = bool(ml.get("golden_cross"))
        ml_score    = float(ml.get("ml_score") or 50)

        # ── Sentiment scores ───────────────────────────────────────────
        news_score  = float(sent.get("sentiment_score") or 0)
        macro_score = float(macro.get("macro_score") or 0)

        # ── Sector tailwinds from macro topics ─────────────────────────
        macro_topics = macro.get("topics") or []
        sector_score = 0
        if macro_topics:
            sector_score = sum(t.get("score", 0) * t.get("weight", 1)
                               for t in macro_topics) / max(len(macro_topics), 1)

        # ══════════════════════════════════════════════════════════════
        # MULTIPLIER ENGINE  (all multipliers clamp to ±20% max)
        # ══════════════════════════════════════════════════════════════
        def clamp(val, lo, hi): return max(lo, min(hi, val))

        # 1. Macro multiplier — broad economy
        macro_mult = clamp(1 + (macro_score / 100) * 0.20, 0.80, 1.20)

        # 2. Sector tailwind multiplier
        sector_mult = clamp(1 + (sector_score / 100) * 0.15, 0.85, 1.15)

        # 3. ROCE trend multiplier — improving ROCE = sustainable growth
        roce_trend  = roce_latest - roce_avg
        if roce_trend > 5:    roce_mult = 1.10
        elif roce_trend > 2:  roce_mult = 1.05
        elif roce_trend > 0:  roce_mult = 1.02
        elif roce_trend > -3: roce_mult = 0.98
        else:                 roce_mult = 0.92

        # 4. Capex/FCF multiplier — company investing in growth
        capex_mult = 1.0
        if fcf_ok:           capex_mult += 0.05
        if ocf and float(ocf) > 0: capex_mult += 0.03
        capex_mult = clamp(capex_mult, 0.95, 1.08)

        # 5. Debt trend multiplier — reducing debt = lower risk
        debt_mult = 1.03 if debt_red else 0.97

        # 6. Promoter holding multiplier
        if promoter > 60:   promo_mult = 1.04
        elif promoter > 50: promo_mult = 1.02
        elif promoter > 40: promo_mult = 1.00
        elif promoter > 25: promo_mult = 0.98
        else:               promo_mult = 0.95

        # 7. News sentiment — dynamic decay based on signal strength
        # Strong signals (war, crisis, major policy) decay slowly
        # Weak signals (minor headlines) decay fast
        news_abs   = abs(news_score) / 100       # 0.0 to 1.0
        decay_rate = 0.50 + news_abs * 0.45      # 0.50 (weak) to 0.95 (strong)

        base_impact = 0.10                        # max 10% impact at 1Y

        w_1y = base_impact * (decay_rate ** 1)
        w_3y = base_impact * (decay_rate ** 3)
        w_5y = base_impact * (decay_rate ** 5)

        direction = news_score / 100              # -1.0 to +1.0

        news_mult_1y = clamp(1 + direction * w_1y, 0.88, 1.12)
        news_mult_3y = clamp(1 + direction * w_3y, 0.94, 1.06)
        news_mult_5y = clamp(1 + direction * w_5y, 0.97, 1.03)

        # 8. RSI & momentum overlay — 1Y price target only
        momentum_mult = 1.0
        if golden and rsi > 55 and ret_3m > 5:   momentum_mult = 1.08
        elif golden and rsi > 50:                 momentum_mult = 1.04
        elif not golden and rsi < 40:             momentum_mult = 0.93
        elif rsi < 30:                            momentum_mult = 0.96
        momentum_mult = clamp(momentum_mult, 0.90, 1.10)

        # ══════════════════════════════════════════════════════════════
        # COMBINED CAGR
        # ══════════════════════════════════════════════════════════════
        base_mult = macro_mult * sector_mult * roce_mult * capex_mult * debt_mult * promo_mult

        # ── Quality-adjusted growth fade ──────────────────────────────
        # Fade rate determined by business quality — strong compounders
        # fade slowly, cyclicals and weak businesses fade aggressively.
        TERMINAL_GROWTH = 10.0

        # ── Step 1: Determine fade rate from quality signals ──────────
        quality_signals = 0

        # Positive signals — slow the fade
        if roce_latest > roce_avg + 2: quality_signals += 1  # ROCE improving
        if fcf_ok:                     quality_signals += 1  # FCF consistently positive
        if debt_red is True:           quality_signals += 1  # reducing debt
        if _opm_trend > 2:             quality_signals += 1  # margins expanding
        if promoter > 55:              quality_signals += 1  # high promoter conviction
        if scr_raw >= 75:              quality_signals += 1  # strong overall fundamentals

        # Negative signals — accelerate the fade
        if _is_metal or _is_infra:     quality_signals -= 2  # cyclical — peak earnings likely
        if _opm_trend < -3:            quality_signals -= 1  # margins declining
        if not fcf_ok:                 quality_signals -= 1  # burning cash
        if _debt_gr > 20 and _sales_1y < 10: quality_signals -= 1  # debt rising, growth not

        # ── Step 2: Map quality signals to fade rate ──────────────────
        if quality_signals >= 4:
            fade_rate = 0.08    # very slow — strong compounder
            label = "compounder"
        elif quality_signals >= 2:
            fade_rate = 0.13    # moderate — good business
            label = "quality"
        elif quality_signals >= 0:
            fade_rate = 0.20    # average — standard fade
            label = "average"
        elif quality_signals >= -2:
            fade_rate = 0.28    # aggressive — weak or cyclical
            label = "cyclical"
        else:
            fade_rate = 0.38    # very aggressive — distressed/peak cycle
            label = "distressed"

        def fade(cagr, years, terminal=TERMINAL_GROWTH):
            """Fade toward a type-specific terminal rate."""
            faded = cagr
            for _ in range(years):
                faded = faded - fade_rate * (faded - terminal)
            return max(faded, terminal * 0.4)

        EPS_TERMINAL    = 12.0
        SALES_TERMINAL  = 9.0
        PROFIT_TERMINAL = 10.0

        eps_1y      = fade(eps_cagr,    1, EPS_TERMINAL)
        eps_3y      = fade(eps_cagr,    3, EPS_TERMINAL)
        eps_5y      = fade(eps_cagr,    5, EPS_TERMINAL)
        sales_1y_f  = fade(sales_cagr,  1, SALES_TERMINAL)
        sales_3y_f  = fade(sales_cagr,  3, SALES_TERMINAL)
        sales_5y_f  = fade(sales_cagr,  5, SALES_TERMINAL)
        profit_1y_f = fade(profit_cagr, 1, PROFIT_TERMINAL)
        profit_3y_f = fade(profit_cagr, 3, PROFIT_TERMINAL)
        profit_5y_f = fade(profit_cagr, 5, PROFIT_TERMINAL)

        # ══════════════════════════════════════════════════════════════
        # PRICE TARGET
        # ══════════════════════════════════════════════════════════════
        def price_target(years, eps_cagr_adj, news_m, mom_m=1.0):
            if not eps or not pe: return None
            fwd_eps  = float(eps) * ((1 + eps_cagr_adj / 100) ** years)
            # Use the lower of current PE or Graham fair PE to avoid inflated targets
            _fair_pe = val_signal.get('fair_pe') if val_signal else None
            exit_pe  = min(float(pe), float(_fair_pe)) if _fair_pe else float(pe)
            exit_pe  = max(exit_pe, 8)  # floor at 8x — no stock priced at zero
            raw      = fwd_eps * exit_pe * news_m * mom_m
            return round(raw, 0)

        pt_1y = price_target(1, eps_1y, news_mult_1y, momentum_mult)
        pt_3y = price_target(3, eps_3y, news_mult_3y)
        pt_5y = price_target(5, eps_5y, news_mult_5y)

        # ══════════════════════════════════════════════════════════════
        # OUTPERFORM CONFIDENCE
        # ══════════════════════════════════════════════════════════════
        scr_score = float(scr_raw)  # use custom score already computed above
        def outperform_prob(years):
            ml_prob    = clamp(ml_score, 30, 80)
            qual_decay = 0.90 if scr_score >= 70 else 0.82 if scr_score >= 50 else 0.75
            prob = ml_prob * (qual_decay ** (years - 1))
            if val_signal:
                pct = float(val_signal.get('pct_vs_fair') or 0)
                if pct > 30:    prob -= 8
                elif pct > 15:  prob -= 4
                elif pct < -20: prob += 5
            return round(clamp(prob, 20, 85), 1)

        # ══════════════════════════════════════════════════════════════
        # FACTOR SUMMARY
        # ══════════════════════════════════════════════════════════════
        def factor_signals():
            ups, downs = [], []
            if macro_mult > 1.05:    ups.append("Positive macro")
            elif macro_mult < 0.95:  downs.append("Negative macro")
            if sector_mult > 1.05:   ups.append("Sector tailwind")
            elif sector_mult < 0.95: downs.append("Sector headwind")
            if roce_mult > 1.05:     ups.append("Improving ROCE")
            elif roce_mult < 0.95:   downs.append("Declining ROCE")
            if capex_mult > 1.05:    ups.append("Strong FCF/Capex")
            if debt_red is True:     ups.append("Debt reducing")
            elif debt_red is False:  downs.append("Debt not reducing")
            if promo_mult > 1.02:    ups.append("High promoter stake")
            elif promo_mult < 0.97:  downs.append("Low promoter stake")
            if news_mult_1y > 1.04:  ups.append("Positive news")
            elif news_mult_1y < 0.96: downs.append("Negative news")
            if momentum_mult > 1.04: ups.append("Strong momentum")
            elif momentum_mult < 0.96: downs.append("Weak momentum")
            return {"up": ups[:3], "down": downs[:3]}

        signals = factor_signals()

        result["forecast"] = {
            "1y": {
                "price_target":       pt_1y,
                "revenue_growth_pct": round(sales_1y_f, 1),
                "profit_growth_pct":  round(profit_1y_f, 1),
                "outperform_prob":    outperform_prob(1),
            },
            "3y": {
                "price_target":       pt_3y,
                "revenue_growth_pct": round(sales_3y_f, 1),
                "profit_growth_pct":  round(profit_3y_f, 1),
                "outperform_prob":    outperform_prob(3),
            },
            "5y": {
                "price_target":       pt_5y,
                "revenue_growth_pct": round(sales_5y_f, 1),
                "profit_growth_pct":  round(profit_5y_f, 1),
                "outperform_prob":    outperform_prob(5),
            },
            "current_price": price_now,
            "signals":       signals,
            "multipliers": {
                "macro":    round(macro_mult, 3),
                "sector":   round(sector_mult, 3),
                "roce":     round(roce_mult, 3),
                "capex":    round(capex_mult, 3),
                "debt":     round(debt_mult, 3),
                "promoter": round(promo_mult, 3),
                "momentum": round(momentum_mult, 3),
                "news_1y":  round(news_mult_1y, 3),
            },
        }
    except Exception:
        result["forecast"] = None

    def fix_nan(obj):
        if isinstance(obj, dict):
            return {k: fix_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_nan(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return jsonify(fix_nan(result))


# ── Peer comparison ───────────────────────────────────────────────────────────
SECTOR_PEERS = {
    'TCS':['INFY','WIPRO','HCLTECH','TECHM','LTM'],
    'INFY':['TCS','WIPRO','HCLTECH','TECHM','LTM'],
    'WIPRO':['TCS','INFY','HCLTECH','TECHM','PERSISTENT'],
    'HCLTECH':['TCS','INFY','WIPRO','TECHM','LTM'],
    'TECHM':['TCS','INFY','WIPRO','HCLTECH','MPHASIS'],
    'LTM':['TCS','INFY','WIPRO','HCLTECH','LTTS'],
    'HDFCBANK':['ICICIBANK','SBIN','KOTAKBANK','AXISBANK','BAJFINANCE'],
    'ICICIBANK':['HDFCBANK','SBIN','KOTAKBANK','AXISBANK','BAJFINANCE'],
    'SBIN':['HDFCBANK','ICICIBANK','KOTAKBANK','BANKBARODA','PNB'],
    'KOTAKBANK':['HDFCBANK','ICICIBANK','SBIN','AXISBANK','BAJFINANCE'],
    'AXISBANK':['HDFCBANK','ICICIBANK','SBIN','KOTAKBANK','BAJFINANCE'],
    'BAJFINANCE':['BAJAJFINSV','CHOLAFIN','MUTHOOTFIN','SHRIRAMFIN','HDFCBANK'],
    'SUNPHARMA':['DRREDDY','CIPLA','LUPIN','DIVISLAB','TORNTPHARM'],
    'DRREDDY':['SUNPHARMA','CIPLA','LUPIN','DIVISLAB','AUROPHARMA'],
    'CIPLA':['SUNPHARMA','DRREDDY','LUPIN','TORNTPHARM','ALKEM'],
    'LUPIN':['SUNPHARMA','DRREDDY','CIPLA','AUROPHARMA','TORNTPHARM'],
    'MARUTI':['M&M','BAJAJ-AUTO','HEROMOTOCO','EICHERMOT','TVSMOTOR'],
    'M&M':['MARUTI','BAJAJ-AUTO','HEROMOTOCO','TVSMOTOR','EICHERMOT'],
    'RELIANCE':['ONGC','BPCL','IOC','HINDPETRO','GAIL'],
    'TATASTEEL':['JSWSTEEL','HINDALCO','VEDL','SAIL','NMDC'],
    'HINDUNILVR':['ITC','NESTLEIND','BRITANNIA','MARICO','DABUR'],
    'ITC':['HINDUNILVR','NESTLEIND','BRITANNIA','MARICO','DABUR'],
    'BRITANNIA':['HINDUNILVR','ITC','NESTLEIND','MARICO','TATACONSUM'],
    'VBL':['BRITANNIA','NESTLEIND','TATACONSUM','MARICO','DABUR'],
    'LT':['SIEMENS','ABB','HAVELLS','CUMMINSIND','POWERGRID'],
    'NTPC':['POWERGRID','TATAPOWER','ADANIGREEN','NHPC','SJVN'],
    'DLF':['GODREJPROP','OBEROIRLTY','PRESTIGE','BRIGADE','SOBHA'],
}

@app.route("/peers")
def peers():
    symbol = request.args.get("symbol","").upper().strip()
    return jsonify({"status":"ok","symbol":symbol,"peers":SECTOR_PEERS.get(symbol,[])[:5]})

@app.route("/compare")
def compare():
    symbols_raw = request.args.get("symbols","")
    if not symbols_raw:
        return jsonify({"error":"symbols required"}),400
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()][:6]

    import math
    results = {}
    lock = threading.Lock()

    def fetch_one(sym):
        try:
            # ── 1. Pull from nightly cache first (fastest, most reliable) ────
            nc   = get_nightly_cache() or {}
            cached = (nc.get('stocks') or {}).get(sym, {})

            # ── 2. Live NSE price ─────────────────────────────────────────────
            q = nse_quote(sym)

            # ── 3. Screener fundamentals CSV ──────────────────────────────────
            fund = {}
            try:
                import pandas as pd
                path    = os.path.join(os.path.dirname(__file__), 'screener_fundamentals.csv')
                sdf     = pd.read_csv(path)
                SYMBOL_CSV_MAP = {'LTM': 'LTIM'}
                csv_sym = SYMBOL_CSV_MAP.get(sym, sym)
                row     = sdf[sdf['symbol'] == csv_sym]
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    fund = {
                        'roce':             r.get('roce_latest_pct'),
                        'sales_cagr_5y':    r.get('sales_cagr_5y'),
                        'profit_cagr_5y':   r.get('profit_cagr_5y'),
                        'investment_score': r.get('investment_score'),
                        'investment_grade': r.get('investment_grade'),
                    }
            except Exception:
                pass

            # ── 4. ML / returns — from nightly cache, then live fallback ──────
            ml = {}
            nc_ml = cached.get('ml') or {}
            if nc_ml.get('ml_score') is not None:
                ml = {
                    'ml_score':   nc_ml.get('ml_score'),
                    'ret_1m_pct': nc_ml.get('ret_1m_pct') or round((cached.get('ret_1m') or 0) * 100, 1),
                    'ret_3m_pct': nc_ml.get('ret_3m_pct') or round((cached.get('ret_3m') or 0) * 100, 1),
                }
            else:
                try:
                    import joblib, pandas as _pd
                    saved = joblib.load(os.path.join(os.path.dirname(__file__), 'ml_model.pkl'))
                    nifty = get_nifty_close()
                    if nifty is not None:
                        f = get_stock_features_cached(sym, nifty)
                        if f:
                            X    = _pd.DataFrame([{k: f[k] for k in saved['features']}])
                            prob = float(saved['model'].predict_proba(X)[0][1])
                            ml   = {
                                'ml_score':   round(prob * 100, 1),
                                'ret_1m_pct': round(f['ret_1m'] * 100, 1),
                                'ret_3m_pct': round(f['ret_3m'] * 100, 1),
                            }
                except Exception:
                    pass

            # ── 5. Valuation — from nightly cache, then yfinance fallback ─────
            val = {}
            nc_val = cached.get('valuation') or cached.get('val') or {}

            pe_nc  = nc_val.get('pe_ratio')
            eps_nc = nc_val.get('eps')
            dy_nc  = nc_val.get('dividend_yield')
            r1y_nc = cached.get('ml', {}).get('ret_1y_pct')
            if r1y_nc is None:
                try:
                    import yfinance as yf
                    info = yf.Ticker(f"{sym}.NS").info
                    raw = info.get('52WeekChange')
                    if raw is not None:
                        val_pct = round(float(raw) * 100, 1)
                        # yfinance sometimes returns already-pct values — sanity cap
                        if -90 <= val_pct <= 300:
                            r1y_nc = val_pct
                except Exception:
                    pass

            if pe_nc or eps_nc:
                val = {
                    'pe_ratio':  pe_nc,
                    'eps':       eps_nc,
                    'div_yield': dy_nc,
                    'ret_1y':    r1y_nc,
                }
            else:
                try:
                    import yfinance as yf
                    info = yf.Ticker(f"{sym}.NS").info
                    val  = {
                        'pe_ratio':  info.get('trailingPE'),
                        'eps':       info.get('trailingEps'),
                        'div_yield': _safe_div_yield(
                            info.get('dividendYield'),
                            info.get('dividendRate'),
                            info.get('currentPrice')
                        ),
                        'ret_1y':    info.get('52WeekChange'),
                    }
                except Exception:
                    pass

            # ── 6. 1Y price target ────────────────────────────────────────────
            pt_1y = None
            try:
                eps     = val.get('eps')
                pe      = float(val.get('pe_ratio') or 20)
                ep_cagr = float(fund.get('profit_cagr_5y') or 8)
                if eps:
                    pt_1y = round(float(eps) * ((1 + ep_cagr / 100) ** 1) * pe, 0)
            except Exception:
                pass

            # ── 7. 1Y return: prefer nightly cache fractional, fallback yfinance
            ret_1y_pct = None
            if r1y_nc is not None:
                try:
                    v = float(r1y_nc)
                    ret_1y_pct = round(v * 100, 1) if abs(v) <= 10 else round(v, 1)
                except Exception:
                    pass
            elif val.get('ret_1y'):
                try:
                    ret_1y_pct = round(float(val['ret_1y']) * 100, 1)
                except Exception:
                    pass

            with lock:
                results[sym] = {
                    'symbol':          sym,
                    'price':           q.get('price'),
                    'change_pct':      q.get('change_pct'),
                    'market_cap':      q.get('market_cap'),
                    'ret_1m':          ml.get('ret_1m_pct'),
                    'ret_3m':          ml.get('ret_3m_pct'),
                    'ret_1y':          ret_1y_pct,
                    'pe_ratio':        val.get('pe_ratio'),
                    'eps':             val.get('eps'),
                    'div_yield':       round(float(val.get('div_yield') or 0), 2) if val.get('div_yield') else None,
                    'roce':            fund.get('roce'),
                    'sales_cagr_5y':   fund.get('sales_cagr_5y'),
                    'profit_cagr_5y':  fund.get('profit_cagr_5y'),
                    'ml_score':        ml.get('ml_score'),
                    'screener_score':  fund.get('investment_score'),
                    'screener_grade':  fund.get('investment_grade'),
                    'combined_grade':  None,
                    'price_target_1y': pt_1y,
                }
        except Exception as e:
            with lock:
                results[sym] = {'symbol': sym, 'error': str(e)}

    threads = [threading.Thread(target=fetch_one,args=(s,)) for s in symbols]
    for t in threads: t.start()
    for t in threads: t.join(timeout=25)

    def fix_nan(obj):
        if isinstance(obj,dict):   return {k:fix_nan(v) for k,v in obj.items()}
        elif isinstance(obj,list): return [fix_nan(v) for v in obj]
        elif isinstance(obj,float) and math.isnan(obj): return None
        return obj

    ordered = [fix_nan(results.get(s,{'symbol':s,'error':'timeout'})) for s in symbols]

    # ── Add relative rankings within peer group ───────────────────────
    RANK_METRICS = {
        'pe_ratio':       {'higher_is': 'bad',  'label': 'P/E Ratio'},
        'roce':           {'higher_is': 'good', 'label': 'ROCE'},
        'sales_cagr_5y':  {'higher_is': 'good', 'label': 'Sales CAGR'},
        'profit_cagr_5y': {'higher_is': 'good', 'label': 'Profit CAGR'},
        'ml_score':       {'higher_is': 'good', 'label': 'ML Score'},
        'screener_score': {'higher_is': 'good', 'label': 'Fundamental Score'},
        'ret_1m':         {'higher_is': 'good', 'label': '1M Return'},
        'ret_3m':         {'higher_is': 'good', 'label': '3M Return'},
        'ret_1y':         {'higher_is': 'good', 'label': '1Y Return'},
        'price_target_1y':{'higher_is': 'good', 'label': '1Y Target'},
    }

    rankings = {}
    for metric, cfg in RANK_METRICS.items():
        vals = [(i, s.get(metric)) for i, s in enumerate(ordered)
                if s.get(metric) is not None and not s.get('error')]
        if len(vals) < 2:
            continue
        reverse = cfg['higher_is'] == 'good'
        sorted_vals = sorted(vals, key=lambda x: x[1], reverse=reverse)
        n = len(sorted_vals)
        for rank, (idx, val) in enumerate(sorted_vals):
            sym = ordered[idx]['symbol']
            if sym not in rankings:
                rankings[sym] = {}
            pct = round((1 - rank / (n - 1)) * 100) if n > 1 else 50
            if pct >= 80:   lbl = 'Best in group'
            elif pct >= 60: lbl = 'Above average'
            elif pct >= 40: lbl = 'Average'
            elif pct >= 20: lbl = 'Below average'
            else:           lbl = 'Worst in group'
            rankings[sym][metric] = {'rank': rank + 1, 'of': n, 'percentile': pct, 'label': lbl}

    # ── Group averages for peer context ──────────────────────────────
    group_avgs = {}
    for metric in ['pe_ratio', 'roce', 'sales_cagr_5y', 'profit_cagr_5y']:
        vals = [s.get(metric) for s in ordered
                if s.get(metric) is not None and not s.get('error')]
        if vals:
            group_avgs[metric] = round(sum(vals) / len(vals), 1)

    for s in ordered:
        sym = s.get('symbol', '')
        s['rankings']   = rankings.get(sym, {})
        s['group_avgs'] = group_avgs

        pe = s.get('pe_ratio')
        avg_pe = group_avgs.get('pe_ratio')
        if pe and avg_pe and avg_pe > 0:
            pe_vs = round((pe - avg_pe) / avg_pe * 100, 1)
            if pe_vs < -20:   s['pe_context'] = f"{abs(pe_vs):.0f}% cheaper than peers"
            elif pe_vs < -5:  s['pe_context'] = f"{abs(pe_vs):.0f}% below peer avg"
            elif pe_vs > 20:  s['pe_context'] = f"{abs(pe_vs):.0f}% more expensive than peers"
            elif pe_vs > 5:   s['pe_context'] = f"{abs(pe_vs):.0f}% above peer avg"
            else:             s['pe_context'] = "In line with peers"

        roce = s.get('roce')
        avg_roce = group_avgs.get('roce')
        if roce and avg_roce and avg_roce > 0:
            roce_vs = round((roce - avg_roce) / avg_roce * 100, 1)
            if roce_vs > 20:    s['roce_context'] = f"{abs(roce_vs):.0f}% above peer avg"
            elif roce_vs < -20: s['roce_context'] = f"{abs(roce_vs):.0f}% below peer avg"
            else:               s['roce_context'] = "In line with peers"

    return jsonify({"status":"ok","count":len(ordered),"data":ordered})


# ── Portfolio vs Nifty + Sector Alerts ───────────────────────────────────────
@app.route("/portfolio-vs-nifty")
def portfolio_vs_nifty():
    raw       = request.args.get("symbols", "")
    from_date = request.args.get("from", "")
    if not raw:
        return jsonify({"error": "symbols required"}), 400

    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:20]

    import yfinance as yf
    from datetime import datetime, timedelta

    # Parse start date
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
    except Exception:
        start = datetime.now() - timedelta(days=30)

    start_str = start.strftime("%Y-%m-%d")
    end_str   = datetime.now().strftime("%Y-%m-%d")

    result = {"portfolio_return": None, "nifty_return": None,
              "beats_nifty": None, "diff": None,
              "stocks": {}, "from_date": start_str}

    try:
        # Nifty return
        nifty = yf.download("^NSEI", start=start_str, end=end_str,
                             auto_adjust=True, progress=False)
        if nifty is not None and len(nifty) >= 2:
            if hasattr(nifty.columns, 'levels'):
                nifty.columns = nifty.columns.get_level_values(0)
            nc = nifty['Close'].squeeze()
            nifty_ret = round((float(nc.iloc[-1]) - float(nc.iloc[0])) / float(nc.iloc[0]) * 100, 2)
            result["nifty_return"] = nifty_ret
    except Exception as e:
        print(f"  [portfolio-vs-nifty] Nifty error: {e}")

    # Each stock's return
    total_w   = 0
    weighted  = 0
    nc = get_nightly_cache() or {}
    for sym in symbols:
        try:
            # Try nightly cache first — much faster
            cached_ret = nc.get('stocks', {}).get(sym, {}).get('ml', {}).get('ret_1y_pct')
            if cached_ret is not None:
                ret = round(float(cached_ret), 2)
            else:
                t = yf.download(f"{sym}.NS", start=start_str, end=end_str,
                                auto_adjust=True, progress=False)
                if t is None or len(t) < 2:
                    result["stocks"][sym] = {"return": None}
                    continue
                if hasattr(t.columns, 'levels'):
                    t.columns = t.columns.get_level_values(0)
                c   = t['Close'].squeeze()
                ret = round((float(c.iloc[-1]) - float(c.iloc[0])) / float(c.iloc[0]) * 100, 2)
            result["stocks"][sym] = {"return": ret}
            weighted += ret
            total_w  += 1
        except Exception:
            result["stocks"][sym] = {"return": None}

    if total_w > 0:
        port_ret = round(weighted / total_w, 2)
        result["portfolio_return"] = port_ret
        if result["nifty_return"] is not None:
            diff = round(port_ret - result["nifty_return"], 2)
            result["diff"]        = diff
            result["beats_nifty"] = diff > 0

    return jsonify(result)


# ── Portfolio X-Ray ───────────────────────────────────────────────────────────
@app.route("/portfolio-xray")
def portfolio_xray():
    raw = request.args.get("symbols", "")
    if not raw:
        return jsonify({"error": "symbols required"}), 400
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:20]

    import math, pandas as pd, threading
    result = {"stocks": {}}

    def _fetch(sym):
        try:
            stock = {}
            # Price from NSE
            q = nse_quote(sym)
            stock['price']  = q.get('price')
            stock['sector'] = q.get('industry', '—')

            # From nightly cache if available
            nc = get_nightly_cache()
            if nc and sym in nc.get('stocks', {}):
                cached = nc['stocks'][sym]
                ml     = cached.get('ml', {})
                val    = cached.get('valuation', {})
                stock['ml_score']       = ml.get('ml_score')
                stock['verdict']        = ml.get('prediction','—').replace('OUTPERFORM','BUY').replace('UNDERPERFORM','HOLD')
                stock['combined_score'] = ml.get('ml_score')  # use ml as proxy if no combined
                # Valuation signal from screener CSV
                try:
                    sdf     = pd.read_csv(os.path.join(os.path.dirname(__file__),'screener_fundamentals.csv'))
                    csv_sym = {'LTM':'LTIM'}.get(sym, sym)
                    row     = sdf[sdf['symbol']==csv_sym]
                    if not row.empty:
                        r          = row.iloc[0].to_dict()
                        scr_score  = float(r.get('investment_score') or 50)
                        stock['combined_score'] = round(ml.get('ml_score',50)*0.5 + scr_score*0.5, 1)
                        stock['verdict']        = 'BUY' if stock['combined_score']>=65 else 'HOLD' if stock['combined_score']>=50 else 'SELL'
                        eps_l = float(r.get('eps_latest') or 0) or None
                        if eps_l and stock['price'] and float(stock['price']) > 0:
                            eps_cagr = float(r.get('eps_cagr_5y') or 8)
                            rbi_rate = RBI_REPO_RATE; rate_adj = 4.4/rbi_rate
                            base_pe  = 22
                            SECTOR_PE= {'IT':28,'Technology':28,'FMCG':35,'Pharma':30,'Banking':15,'Finance':18,'Metals':10,'Energy':12,'Defence':30}
                            for k,v in SECTOR_PE.items():
                                if k.lower() in str(stock.get('sector','')).lower():
                                    base_pe=v; break
                            qm = 1.0
                            if float(r.get('roce_latest_pct') or 10)>float(r.get('roce_avg_5y') or 10)+3: qm+=0.10
                            if bool(r.get('fcf_positive_3y')): qm+=0.08
                            if bool(r.get('debt_reducing')):   qm+=0.07
                            qm = min(qm, 1.35)
                            fair_pe  = round(min((base_pe+1.5*min(eps_cagr,25))*rate_adj*qm, 55),1)
                            fair_val = round(eps_l * fair_pe, 0)
                            cur_p    = float(stock['price'])
                            pct_fair = round((cur_p - fair_val)/fair_val*100,1) if fair_val>0 else 0
                            mos      = 0.10 if scr_score>=75 else 0.15 if scr_score>=60 else 0.25
                            stock['fair_value']   = fair_val
                            stock['val_signal']   = ('Undervalued Quality' if scr_score>=60 and cur_p<fair_val*0.90
                                                     else 'Overvalued Quality' if scr_score>=60 and cur_p>fair_val*1.15
                                                     else 'Fairly Valued' if scr_score>=60
                                                     else 'Value Trap Risk' if cur_p<fair_val*0.90
                                                     else 'Overpriced')
                            stock['val_signal_color'] = ('green' if 'Undervalued' in stock['val_signal']
                                                         else 'red' if 'Trap' in stock['val_signal'] or 'Overpriced' in stock['val_signal']
                                                         else 'gold')
                except Exception:
                    pass
            result['stocks'][sym] = {k:v for k,v in stock.items()
                                      if v is not None and not (isinstance(v,float) and math.isnan(v))}
        except Exception as e:
            result['stocks'][sym] = {'error': str(e)}

    threads = [threading.Thread(target=_fetch, args=(sym,)) for sym in symbols]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15)

    return jsonify({"status":"ok","stocks":result["stocks"]})


# ── PDF Report Generator ──────────────────────────────────────────────────────
@app.route("/generate-report")
def generate_report_endpoint():
    symbol    = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    # Accept pre-computed values from frontend to ensure score consistency
    _score    = request.args.get("score")
    _score_10 = request.args.get("score_10")
    _grade    = request.args.get("grade")
    try:
        from report_generator import generate_report
        from flask import Response
        import math, threading

        # Fetch all data in parallel — same pattern as stock_analysis
        data = {"symbol": symbol, "status": "ok"}

        def _quote():
            try: data["quote"] = nse_quote(symbol)
            except Exception: data["quote"] = {}

        def _ml():
            try:
                nc = get_nightly_cache()
                if nc and symbol in nc.get('stocks', {}):
                    cached = nc['stocks'][symbol]
                    data['ml']              = cached['ml']
                    data['_val_from_cache'] = cached.get('valuation')
                    data['chart_insights']  = cached.get('chart_insights', {})
                    return
            except Exception:
                pass
            try:
                import joblib, pandas as pd, yfinance as _yf, numpy as np
                saved    = joblib.load(os.path.join(os.path.dirname(__file__), 'ml_model.pkl'))
                model    = saved['model']; features = saved['features']; accuracy = saved['accuracy']
                stock_df = _yf.download(f"{symbol}.NS", period="2y", interval="1d",
                                        auto_adjust=True, progress=False)
                nc = get_nifty_close()
                nifty_df = pd.DataFrame({'Close': nc}, index=nc.index) if nc is not None and len(nc) >= 100 \
                           else _yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False)
                if stock_df is None or len(stock_df) < 100:
                    data["ml"] = {"error": "Insufficient history"}; return
                if hasattr(stock_df.columns, 'levels'): stock_df.columns = stock_df.columns.get_level_values(0)
                if hasattr(nifty_df.columns, 'levels'): nifty_df.columns = nifty_df.columns.get_level_values(0)
                sw_s = pd.Series(stock_df['Close'].squeeze().values, index=stock_df.index, dtype=float)
                nw_s = pd.Series(nifty_df['Close'].squeeze().values, index=nifty_df.index, dtype=float)
                common = sw_s.index.intersection(nw_s.index)
                sw = sw_s.loc[common].values.astype(float) if len(common) >= 100 else sw_s.values.astype(float)
                nw = nw_s.loc[common].values.astype(float) if len(common) >= 100 else None
                cp = float(sw[-1])
                def sr(a, b): return float(a[-1]/a[-b]-1) if len(a) > b and a[-b] > 0 else 0.0
                ret_1m=sr(sw,22); ret_3m=sr(sw,63); ret_6m=sr(sw,126); ret_1y=sr(sw,min(200,len(sw)-1))
                ma50=float(np.mean(sw[-min(50,len(sw)):])); ma200=float(np.mean(sw[-min(200,len(sw)):]))
                dr=np.diff(sw)/sw[:-1]; dr=dr[~np.isnan(dr)]
                vol_1m=float(np.std(dr[-22:])*np.sqrt(252)) if len(dr)>=22 else 0.3
                vol_3m=float(np.std(dr[-63:])*np.sqrt(252)) if len(dr)>=63 else 0.3
                h52=float(np.max(sw[-252:])) if len(sw)>=252 else float(np.max(sw))
                l52=float(np.min(sw[-252:])) if len(sw)>=252 else float(np.min(sw))
                rng=h52-l52
                d_r=np.diff(sw[-15:]) if len(sw)>=15 else np.array([0.001,-0.001])
                g=d_r[d_r>0].mean() if len(d_r[d_r>0])>0 else 0.001
                ls=abs(d_r[d_r<0].mean()) if len(d_r[d_r<0])>0 else 0.001
                f = {
                    'ret_1m':ret_1m,'ret_3m':ret_3m,'ret_6m':ret_6m,'ret_1y':ret_1y,
                    'rs_1m':ret_1m-sr(nw,22) if nw is not None else 0.0,
                    'rs_3m':ret_3m-sr(nw,63) if nw is not None else 0.0,
                    'price_to_ma50':cp/ma50-1 if ma50>0 else 0,
                    'price_to_ma200':cp/ma200-1 if ma200>0 else 0,
                    'golden_cross':1 if ma50>ma200 else 0,
                    'vol_1m':vol_1m,'vol_3m':vol_3m,
                    'pos52':float((cp-l52)/rng) if rng>0 else 0.5,
                    'rsi':float(100-100/(1+g/ls)),
                    'vol_trend':float(vol_1m/vol_3m) if vol_3m>0 else 1.0,
                    'roce_latest_pct':12.0,'opm_latest_pct':12.0,'sales_cagr_5y':10.0,
                    'profit_cagr_5y':8.0,'eps_cagr_5y':8.0,'sales_growth_1y':8.0,
                    'profit_growth_1y':8.0,'opm_trend_5y':0.0,'roce_trend_5y':0.0,
                    'promoter_pct':45.0,'fii_pct':15.0,'fcf_positive_3y':0.5,
                    'debt_reducing':0.5,'screener_de':50.0,
                    'pe_ratio':22.0,'pb_ratio':3.0,'peg_ratio':2.5,
                }
                try:
                    sdf=pd.read_csv(os.path.join(os.path.dirname(__file__),'screener_fundamentals.csv'))
                    csv_sym={'LTM':'LTIM'}.get(symbol,symbol)
                    row=sdf[sdf['symbol']==csv_sym]
                    r=row.iloc[0].to_dict() if not row.empty else {}
                    for k, d in [('roce_latest_pct',12.0),('opm_latest_pct',12.0),('sales_cagr_5y',10.0),
                                  ('profit_cagr_5y',8.0),('eps_cagr_5y',8.0),('sales_growth_1y',8.0),
                                  ('profit_growth_1y',8.0),('opm_trend_5y',0.0),('roce_trend_5y',0.0),
                                  ('promoter_pct',45.0),('fii_pct',15.0),('fcf_positive_3y',0.5),
                                  ('debt_reducing',0.5),('screener_de',50.0)]:
                        try: f[k]=float(r.get(k) or d)
                        except: f[k]=d
                    eps_l=float(r.get('eps_latest') or 0) or None
                    if eps_l and eps_l > 0 and cp > 0:
                        f['pe_ratio']=round(cp/eps_l,1)
                        f['peg_ratio']=round(f['pe_ratio']/max(f['eps_cagr_5y'],0.1),2)
                except Exception: pass
                X=pd.DataFrame([{k:f[k] for k in features}])
                prob=float(model.predict_proba(X)[0][1]); pred=int(model.predict(X)[0])
                data["ml"]={
                    "ml_score":round(prob*100,1),"prediction":"OUTPERFORM" if pred==1 else "UNDERPERFORM",
                    "accuracy":round(accuracy*100,1),"rsi":round(f['rsi'],1),
                    "pos52_pct":round(f['pos52']*100,1),"ret_1m_pct":round(ret_1m*100,1),
                    "ret_3m_pct":round(ret_3m*100,1),"golden_cross":bool(f['golden_cross']),
                }
            except Exception as e:
                data["ml"] = {"error": str(e)}

        def _fund():
            try:
                import pandas as pd
                sdf=pd.read_csv(os.path.join(os.path.dirname(__file__),'screener_fundamentals.csv'))
                csv_sym={'LTM':'LTIM'}.get(symbol,symbol)
                row=sdf[sdf['symbol']==csv_sym]
                if not row.empty:
                    r=row.iloc[0].to_dict()
                    data["fundamentals"]={
                        "roce":               r.get('roce_latest_pct'),
                        "roce_latest_pct":    r.get('roce_latest_pct'),
                        "roce_avg_5y":        r.get('roce_avg_5y'),
                        "roce_trend_5y":      r.get('roce_trend_5y'),
                        "sales_cagr_5y":      r.get('sales_cagr_5y'),
                        "sales_cagr_10y":     r.get('sales_cagr_10y'),
                        "sales_growth_1y":    r.get('sales_growth_1y'),
                        "profit_cagr_5y":     r.get('profit_cagr_5y'),
                        "profit_cagr_10y":    r.get('profit_cagr_10y'),
                        "profit_growth_1y":   r.get('profit_growth_1y'),
                        "eps_cagr_5y":        r.get('eps_cagr_5y'),
                        "eps_growth_1y":      r.get('eps_growth_1y'),
                        "opm_latest_pct":     r.get('opm_latest_pct'),
                        "opm_avg_5y":         r.get('opm_avg_5y'),
                        "opm_trend_5y":       r.get('opm_trend_5y'),
                        "fcf_positive_3y":    r.get('fcf_positive_3y'),
                        "ocf_positive_3y":    r.get('ocf_positive_3y'),
                        "fcf_cagr_5y":        r.get('fcf_cagr_5y'),
                        "ocf_latest_cr":      r.get('ocf_latest_cr'),
                        "debt_reducing":      r.get('debt_reducing'),
                        "debt_growth_1y":     r.get('debt_growth_1y'),
                        "screener_de":        r.get('screener_de'),
                        "networth_cr":        r.get('networth_cr'),
                        "promoter_pct":       r.get('promoter_pct'),
                        "fii_pct":            r.get('fii_pct'),
                        "dii_pct":            r.get('dii_pct'),
                        "eps_latest":         r.get('eps_latest'),
                        "dividend_payout_pct":r.get('dividend_payout_pct'),
                        "profit_latest_cr":   r.get('profit_latest_cr'),
                        "sales_latest_cr":    r.get('sales_latest_cr'),
                        "investment_score":   r.get('investment_score'),
                        "investment_grade":   r.get('investment_grade'),
                    }
                else: data["fundamentals"]={}
            except Exception: data["fundamentals"]={}

        def _val():
            cached_val = data.get('_val_from_cache')
            if cached_val:
                data['valuation'] = cached_val; return
            try:
                import yfinance as yf, concurrent.futures, pandas as pd
                t=yf.Ticker(f"{symbol}.NS"); fi=t.fast_info
                price=getattr(fi,'last_price',None); shares=getattr(fi,'shares',None)
                def _eps():
                    try:
                        inc=t.income_stmt
                        if inc is not None and not inc.empty:
                            for k in inc.index:
                                if 'net income' in str(k).lower():
                                    ni=float(inc.loc[k].iloc[0])
                                    return round(ni/float(shares),2) if shares and float(shares)>0 else None
                    except Exception: pass
                def _div():
                    try:
                        divs=t.dividends
                        if divs is not None and len(divs)>0:
                            cutoff=pd.Timestamp.now(tz=divs.index.tz)
                            one_yr=divs[divs.index>=(cutoff-pd.DateOffset(years=1))]
                            annual=float(one_yr.sum())
                            return round(annual/float(price),6) if annual>0 and price and float(price)>0 else None
                    except Exception: pass
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    fe=ex.submit(_eps); fd=ex.submit(_div)
                    try: eps=fe.result(timeout=6)
                    except Exception: eps=None
                    try: div_yield=fd.result(timeout=6)
                    except Exception: div_yield=None
                pe=round(float(price)/float(eps),1) if price and eps and float(eps)>0 else None
                data["valuation"]={"pe_ratio":pe,"eps":eps,"dividend_yield":div_yield,
                                    "pb_ratio":None,"profit_margin":None,"revenue_growth":None,
                                    "earnings_growth":None,"debt_to_equity":None}
            except Exception: data["valuation"]={}

        def _sent():
            try:
                from news_sentiment import get_sentiment_score
                data["sentiment"]=get_sentiment_score(symbol)
            except Exception: data["sentiment"]={"sentiment_score":0,"sentiment_label":"neutral"}

        def _macro():
            try:
                from ml_screener import _cache as _mc
                from macro_sentiment import get_macro_sentiment, apply_macro_to_stock as _apply
                mc=_mc.get('macro_news',{})
                macro_data=mc.get('data') if mc.get('data') and mc.get('ts',0)>0 else get_macro_sentiment()
                data["macro"]=_apply(symbol,macro_data)
            except Exception: data["macro"]={"macro_score":0,"macro_label":"neutral"}

        threads=[threading.Thread(target=f) for f in [_quote,_ml,_fund,_val,_sent,_macro]]
        for t in threads: t.start()
        for t in threads: t.join(timeout=35)

        # Build combined score (same logic as stock_analysis)
        try:
            # ── If frontend passed a pre-computed score, use it directly ──
            if _score:
                try:
                    combined = float(_score)
                    if combined >= 82:   verdict, verdict_color = 'STRONG BUY', 'green'
                    elif combined >= 68: verdict, verdict_color = 'BUY',        'green'
                    elif combined >= 58: verdict, verdict_color = 'MILD BUY',   'green'
                    elif combined >= 48: verdict, verdict_color = 'HOLD',       'gold'
                    elif combined >= 38: verdict, verdict_color = 'MILD SELL',  'red'
                    else:                verdict, verdict_color = 'SELL',       'red'
                    grade = 'A+' if combined>=80 else 'A' if combined>=70 else 'B' if combined>=60 else 'C' if combined>=50 else 'D'
                except Exception:
                    pass
            ml_raw=data.get("ml",{}).get("ml_score",50) or 50
            scr_raw=data.get("fundamentals",{}).get("custom_score") or data.get("fundamentals",{}).get("investment_score",50) or 50
            pe=data.get("valuation",{}).get("pe_ratio") or 20
            yfin_score=50
            if pe<12: yfin_score+=12
            elif pe<18: yfin_score+=7
            elif pe<25: yfin_score+=3
            elif pe>40: yfin_score-=8
            # PEG adjustment in report endpoint
            try:
                _fund_rep = data.get('fundamentals', {})
                _eps_cagr_rep = float(_fund_rep.get('eps_cagr_5y') or 0)
                if _eps_cagr_rep > 0 and pe > 0:
                    _peg_rep = pe / _eps_cagr_rep
                    if _peg_rep < 0.5:    yfin_score += 15
                    elif _peg_rep < 0.8:  yfin_score += 10
                    elif _peg_rep < 1.2:  yfin_score += 5
                    elif _peg_rep < 2.0:  yfin_score -= 2
                    elif _peg_rep < 3.0:  yfin_score -= 6
                    else:                 yfin_score -= 12
            except Exception:
                pass
            # ROCE bonus
            _roce_r = float(data.get('fundamentals',{}).get('roce') or data.get('fundamentals',{}).get('roce_latest_pct') or 0)
            if _roce_r > 25:   yfin_score += 8
            elif _roce_r > 15: yfin_score += 4
            elif _roce_r < 8:  yfin_score -= 8
            yfin_score=max(0,min(100,yfin_score))
            sent_raw=data.get("sentiment",{}).get("sentiment_score",0) or 0
            sent_score=max(0,min(100,50+sent_raw*0.5))
            macro_raw=data.get("macro",{}).get("macro_score",0) or 0
            macro_score=max(0,min(100,50+macro_raw*0.5))
            combined=round(ml_raw*0.22+scr_raw*0.44+yfin_score*0.34,1)
            grade='A+' if combined>=80 else 'A' if combined>=70 else 'B' if combined>=60 else 'C' if combined>=50 else 'D'
            if combined >= 82:   verdict, verdict_color = 'STRONG BUY', 'green'
            elif combined >= 68: verdict, verdict_color = 'BUY',        'green'
            elif combined >= 58: verdict, verdict_color = 'MILD BUY',   'green'
            elif combined >= 48: verdict, verdict_color = 'HOLD',       'gold'
            elif combined >= 38: verdict, verdict_color = 'MILD SELL',  'red'
            else:                verdict, verdict_color = 'SELL',       'red'
            rsi_val=float(data.get('ml',{}).get('rsi') or 50)
            pos52=float(data.get('ml',{}).get('pos52_pct') or 50)
            ret_1m=float(data.get('ml',{}).get('ret_1m_pct') or 0)
            if sent_raw<-15 and scr_raw>=60 and rsi_val<45 and pos52<40 and ret_1m<-5 and combined>=42:
                verdict='BUY' if scr_raw<75 else 'STRONG BUY'; verdict_color='green'
            mcap_str=data.get('quote',{}).get('market_cap','') or ''
            try:
                mc_val=float(mcap_str.replace('₹','').replace('L Cr','').replace('T Cr','').replace('Cr','').replace(',','').strip() or 0)
                is_large='L Cr' in mcap_str and mc_val>50
            except Exception: is_large=False
            _xr_fund   = data.get('fundamentals', {})
            _xr_sec    = str(data.get('quote', {}).get('industry', '') or '').lower()
            _xr_bank   = any(x in _xr_sec for x in ['bank','nbfc','financ','insurance','microfinance'])
            _xr_metal  = any(x in _xr_sec for x in ['metal','steel','alumin','mining'])
            _xr_de     = float(_xr_fund.get('screener_de') or 0) or None
            _xr_debt_r = str(_xr_fund.get('debt_reducing','')).lower() == 'true'
            _xr_opm_t  = float(_xr_fund.get('opm_trend_5y') or 0)
            _xr_opm_l  = float(_xr_fund.get('opm_latest_pct') or 0)
            _xr_fcf    = str(_xr_fund.get('fcf_positive_3y','')).lower() == 'true'
            _xr_s1y    = float(_xr_fund.get('sales_growth_1y') or 0)
            _xr_p1y    = float(_xr_fund.get('profit_growth_1y') or 0)
            _xr_sc5    = float(_xr_fund.get('sales_cagr_5y') or 0)
            _xr_pc5    = float(_xr_fund.get('profit_cagr_5y') or 0)
            vol        = abs(float(data.get('ml',{}).get('ret_1m_pct') or 0))
            rsi_xr     = float(data.get('ml',{}).get('rsi') or 50)
            p52_xr     = float(data.get('ml',{}).get('pos52_pct') or 50)

            xr_risk_pts = 0
            if not is_large:                xr_risk_pts += 2
            if _xr_bank:                    xr_risk_pts += 0
            elif _xr_debt_r:                xr_risk_pts += 0
            elif _xr_de is None or _xr_de < 0.3: xr_risk_pts += 0
            elif _xr_de < 0.8:              xr_risk_pts += 1 if _xr_s1y <= 15 else 0
            elif _xr_de < 1.5:              xr_risk_pts += 2
            else:                           xr_risk_pts += 3
            if not _xr_bank:
                if _xr_opm_t < -5:          xr_risk_pts += 2
                elif _xr_opm_t < -2:        xr_risk_pts += 1
                if _xr_opm_l < 0:           xr_risk_pts += 2
            if not _xr_fcf:                 xr_risk_pts += 1
            if vol < 5:                     xr_risk_pts += 0
            elif vol < 12:                  xr_risk_pts += 1
            else:                           xr_risk_pts += 2
            if rsi_xr > 78:                 xr_risk_pts += 1
            elif rsi_xr < 28:               xr_risk_pts += 1
            if p52_xr < 15:                 xr_risk_pts += 1
            if scr_raw < 45:                xr_risk_pts += 2
            elif scr_raw >= 70:             xr_risk_pts -= 1
            if _xr_sc5 < 0 and _xr_pc5 < 0: xr_risk_pts += 2
            elif _xr_sc5 > 15 and _xr_pc5 > 15: xr_risk_pts -= 1
            xr_risk_pts = max(0, xr_risk_pts)
            risk = 'Low' if xr_risk_pts <= 2 else 'Medium' if xr_risk_pts <= 5 else 'High'
            reasons=[]
            if scr_raw>=70: reasons.append('Strong fundamentals')
            elif scr_raw<45: reasons.append('Weak fundamentals')
            if ml_raw>=60: reasons.append('positive ML signal')
            elif ml_raw<40: reasons.append('negative ML signal')
            if sent_raw>10: reasons.append('positive news')
            elif sent_raw<-10: reasons.append('negative news')
            reason=reasons[0].capitalize()+(', '+reasons[1] if len(reasons)>1 else '') if reasons else 'Mixed signals'

            # Valuation signal
            val_signal=None
            try:
                import pandas as pd
                sdf=pd.read_csv(os.path.join(os.path.dirname(__file__),'screener_fundamentals.csv'))
                csv_sym={'LTM':'LTIM'}.get(symbol,symbol)
                row=sdf[sdf['symbol']==csv_sym]
                r=row.iloc[0].to_dict() if not row.empty else {}
                eps_latest=float(r.get('eps_latest') or 0)
                eps_cagr_v=float(r.get('eps_cagr_5y') or 8)
                cur_price=float(str(data.get('quote',{}).get('price') or 0).replace(',','')) or None
                cur_pe=float(data.get('valuation',{}).get('pe_ratio') or 0)
                if eps_latest>0 and cur_price:
                    rbi_rate   = 6.5
                    rate_adj   = 4.4 / rbi_rate
                    SECTOR_PE  = {
                        'IT':28,'Technology':28,'Software':28,'FMCG':35,'Consumer':32,
                        'Pharma':30,'Healthcare':30,'Banking':15,'Finance':18,'NBFC':18,
                        'Auto':20,'Automobile':20,'Metals':10,'Steel':10,'Mining':10,
                        'Energy':12,'Oil':12,'Power':14,'Infrastructure':18,
                        'Construction':16,'Cement':20,'Real Estate':20,'Defence':30,
                    }
                    sector_str = str(data.get('quote',{}).get('industry','') or '')
                    base_pe    = 22
                    for k,v in SECTOR_PE.items():
                        if k.lower() in sector_str.lower():
                            base_pe = v; break
                    quality_mult = 1.0
                    if float(r.get('roce_latest_pct') or 10) > float(r.get('roce_avg_5y') or 10)+3: quality_mult+=0.10
                    if bool(r.get('fcf_positive_3y')): quality_mult+=0.08
                    if bool(r.get('debt_reducing')):   quality_mult+=0.07
                    if float(r.get('promoter_pct') or 0)>55: quality_mult+=0.05
                    quality_mult = min(quality_mult, 1.35)
                    reliable_growth = min(float(eps_cagr_v or 8.0), 25)
                    if float(r.get('opm_trend_5y') or 0) < -3: reliable_growth = min(reliable_growth, 15)
                    fair_pe = round(min((base_pe+1.5*reliable_growth)*rate_adj*quality_mult, 55), 1)
                    fair_value=round(eps_latest*fair_pe,1)
                    mos=0.10 if scr_raw>=75 else 0.15 if scr_raw>=60 else 0.25
                    buy_zone_high=round(fair_value*(1-mos*0.5),1)
                    buy_zone_low=round(fair_value*(1-mos),1)
                    pct_vs_fair=round((cur_price-fair_value)/fair_value*100,1) if fair_value>0 else 0
                    confidence=50
                    if eps_cagr_v>15: confidence+=15
                    elif eps_cagr_v>10: confidence+=10
                    elif eps_cagr_v>5: confidence+=5
                    if float(r.get('roce_latest_pct') or 0)>float(r.get('roce_avg_5y') or 0): confidence+=10
                    if r.get('fcf_positive_3y'): confidence+=10
                    if r.get('debt_reducing'): confidence+=5
                    if scr_raw>=75: confidence+=10
                    elif scr_raw<45: confidence-=10
                    confidence=max(20,min(90,confidence))
                    is_quality=scr_raw>=60
                    is_expensive=cur_price>fair_value*1.15
                    is_cheap=cur_price<fair_value*0.90
                    if is_quality and is_cheap: sig_label,sig_color='Undervalued Quality','green'
                    elif is_quality and is_expensive: sig_label,sig_color='Overvalued Quality','gold'
                    elif is_quality: sig_label,sig_color='Fairly Valued Quality','green'
                    elif is_cheap: sig_label,sig_color='Value Trap Risk','red'
                    else: sig_label,sig_color='Overpriced Weak Business','red'
                    val_signal={'label':sig_label,'color':sig_color,'fair_value':fair_value,
                                'fair_pe':round(fair_pe,1),'current_pe':cur_pe if cur_pe>0 else None,
                                'pct_vs_fair':pct_vs_fair,'buy_zone_low':buy_zone_low,
                                'buy_zone_high':buy_zone_high,'confidence':confidence,'current_price':cur_price,
                                'description': 'Strong business trading below fair value — opportunity' if sig_label=='Undervalued Quality'
                                               else 'Strong business but priced above fair value — wait for dip' if sig_label=='Overvalued Quality'
                                               else 'Strong business at a fair price' if sig_label=='Fairly Valued Quality'
                                               else 'Cheap valuation but weak fundamentals — be cautious' if sig_label=='Value Trap Risk'
                                               else 'Weak fundamentals and expensive — avoid'}
            except Exception: val_signal=None

            data["combined"]={
                "score":    float(_score)    if _score    else combined,
                "score_10": float(_score_10) if _score_10 else round(combined/10,1),
                "verdict":  verdict,
                "grade":    _grade           if _grade    else grade,
                "yfin_score":round(yfin_score,1),
                "sent_score":round(sent_score,1),"macro_score":round(macro_score,1),
                "screener_score":round(scr_raw,1),"verdict":verdict,"verdict_color":verdict_color,
                "risk":risk,"risk_color":'green' if risk=='Low' else 'gold' if risk=='Medium' else 'red',
                "reason":reason,"score_10":round(combined/10,1),"valuation_signal":val_signal,
            }
        except Exception: data["combined"]={"score":50,"grade":"C","verdict":"HOLD"}

        # Build forecast (simplified)
        try:
            fund=data.get("fundamentals") or {}
            quote_d=data.get("quote") or {}
            ml_d=data.get("ml") or {}
            sent_d=data.get("sentiment") or {}
            macro_d=data.get("macro") or {}
            pe_v=float(data.get("valuation",{}).get("pe_ratio") or 20)
            price_now=float(str(quote_d.get("price") or 0).replace(",","")) or None
            sales_cagr=min(float(fund.get("sales_cagr_5y") or 8),25)
            profit_cagr=min(float(fund.get("profit_cagr_5y") or 8),30)
            eps_cagr_f=min(float(fund.get("eps_cagr_5y") or profit_cagr),30)
            ml_s=float(ml_d.get("ml_score") or 50)
            news_s=float(sent_d.get("sentiment_score") or 0)
            macro_s=float(macro_d.get("macro_score") or 0)
            macro_mult=max(0.80,min(1.20,1+(macro_s/100)*0.20))
            roce_l=float(fund.get("roce") or 10); roce_a=float(fund.get("roce_avg_5y") or roce_l)
            roce_mult=1.10 if roce_l-roce_a>5 else 1.05 if roce_l-roce_a>2 else 1.02 if roce_l-roce_a>0 else 0.98 if roce_l-roce_a>-3 else 0.92
            fcf_ok=fund.get("fcf_positive_3y"); debt_red=fund.get("debt_reducing")
            capex_mult=1.0+(0.05 if fcf_ok else 0)+(0.03 if debt_red else 0)
            momentum_mult=max(0.85,min(1.15,1+(float(ml_d.get("ret_3m_pct") or 0)/100)*0.3))
            news_mult=max(0.90,min(1.10,1+(news_s/100)*0.10))
            prom=float(fund.get("promoter_pct") or 40)
            promoter_mult=1.05 if prom>60 else 1.02 if prom>45 else 0.97 if prom<30 else 1.0
            base_out=ml_s/100
            # ── Quality-adjusted fade (same as stock_analysis) ─────────
            _r_scr_rep = data.get('fundamentals', {})
            _r_roce_rep = float(_r_scr_rep.get('roce') or _r_scr_rep.get('roce_latest_pct') or 10)
            _r_roce_avg_rep = float(_r_scr_rep.get('roce_avg_5y') or _r_roce_rep)
            _r_fcf_rep  = str(_r_scr_rep.get('fcf_positive_3y','')).lower() == 'true'
            _r_debt_rep = str(_r_scr_rep.get('debt_reducing','')).lower() == 'true'
            _r_opm_t_rep= float(_r_scr_rep.get('opm_trend_5y') or 0)
            _r_prom_rep = float(_r_scr_rep.get('promoter_pct') or 40)
            _r_sec_rep  = str(data.get('quote',{}).get('industry','') or '').lower()
            _r_metal_rep= any(x in _r_sec_rep for x in ['metal','steel','alumin','mining','oil','petro','refin','fertiliser','chemical','coal','gas','cement'])

            qs = 0
            if _r_roce_rep > _r_roce_avg_rep + 2:  qs += 1
            if _r_fcf_rep:                          qs += 1
            if _r_debt_rep:                         qs += 1
            if _r_opm_t_rep > 2:                    qs += 1
            if _r_prom_rep > 55:                    qs += 1
            if _r_metal_rep:                        qs -= 2
            if _r_opm_t_rep < -3:                   qs -= 1
            if not _r_fcf_rep:                      qs -= 1

            _fr = 0.08 if qs >= 4 else 0.13 if qs >= 2 else 0.20 if qs >= 0 else 0.28 if qs >= -2 else 0.38
            TERM = 10.0

            def _fade_rep(cagr, yrs, term=TERM):
                v = cagr
                for _ in range(yrs):
                    v = v - _fr * (v - term)
                return max(v, term * 0.4)

            eps_rep  = data.get('valuation', {}).get('eps')
            pe_rep   = float(data.get('valuation', {}).get('pe_ratio') or 20)
            _fpe_rep = data.get('combined', {}).get('valuation_signal') or {}
            _fpe_rep = float(_fpe_rep.get('fair_pe') or pe_rep) if _fpe_rep else pe_rep
            exit_pe_rep = max(min(pe_rep, _fpe_rep), 8)

            def proj(yrs, sc, pc, ec):
                rev_f  = round(_fade_rep(sc, yrs, 9.0), 1)
                prof_f = round(_fade_rep(pc, yrs, 10.0), 1)
                eps_f  = round(_fade_rep(ec, yrs, 12.0), 1)
                pt = None
                if eps_rep and pe_rep:
                    fwd_eps = float(eps_rep) * ((1 + eps_f / 100) ** yrs)
                    pt = round(fwd_eps * exit_pe_rep * macro_mult * roce_mult * capex_mult * news_mult * promoter_mult, 0)
                prob = min(75, max(35, ml_s + (scr_raw - 50) * 0.2 - yrs * 2.0))
                if price_now and pt:
                    upside = (pt - price_now) / price_now * 100
                    if upside < 0: prob = max(prob - 10, 20)
                return {
                    "price_target":       pt,
                    "revenue_growth_pct": rev_f,
                    "profit_growth_pct":  prof_f,
                    "outperform_prob":    round(prob, 1)
                }

            data["forecast"] = {
                "current_price": price_now,
                "1y": proj(1, sales_cagr, profit_cagr, eps_cagr_f),
                "3y": proj(3, sales_cagr, profit_cagr, eps_cagr_f),
                "5y": proj(5, sales_cagr, profit_cagr, eps_cagr_f),
            }
        except Exception: data["forecast"]={}

        def fix_nan(obj):
            if isinstance(obj,dict): return {k:fix_nan(v) for k,v in obj.items()}
            elif isinstance(obj,list): return [fix_nan(v) for v in obj]
            elif isinstance(obj,float) and math.isnan(obj): return None
            return obj

        pdf_bytes = generate_report(fix_nan(data))
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{symbol}_Investment_Report.pdf"',
                'Content-Length': str(len(pdf_bytes)),
                'Access-Control-Expose-Headers': 'Content-Disposition',
            }
        )
    except Exception as e:
        import traceback
        print(f"[report ERROR] {traceback.format_exc()}", flush=True)
        return jsonify({"error": str(e)}), 500


# ── Macro sentiment ───────────────────────────────────────────────────────────
@app.route("/macro-sentiment")
def macro_sentiment():
    """Return cached macro sentiment topics for Sector Rotation Alerts."""
    try:
        from macro_sentiment import get_macro_sentiment
        from ml_screener import _cache
        mc = _cache.get('macro_news', {})
        if mc.get('data') and (time.time() - mc.get('ts', 0)) < 3600:
            macro_data = mc['data']
        else:
            macro_data = get_macro_sentiment()
            _cache['macro_news'] = {'data': macro_data, 'ts': time.time()}

        topics = []
        for item in macro_data:
            topics.append({
                'topic':     item.get('topic', ''),
                'label':     item.get('sentiment', 'neutral'),
                'sentiment': item.get('sentiment', 'neutral'),
                'score':     item.get('score', 0),
            })

        return jsonify({"status": "ok", "topics": topics})
    except Exception as e:
        return jsonify({"status": "error", "topics": [], "message": str(e)})


# ── Nightly cache rebuild trigger ─────────────────────────────────────────────
@app.route("/rebuild-cache")
def rebuild_cache():
    secret = request.args.get("secret", "")
    if secret != os.environ.get("CACHE_SECRET", "graham2024"):
        return jsonify({"error": "unauthorized"}), 401
    def _rebuild():
        try:
            import nightly_cache
            print("  [rebuild] Starting nightly cache rebuild...", flush=True)
            nightly_cache.build_cache()
            print("  [rebuild] Cache rebuild complete.", flush=True)
        except Exception as e:
            print(f"  [rebuild] Error: {e}", flush=True)
    threading.Thread(target=_rebuild, daemon=True).start()
    return jsonify({"status": "ok", "message": "Cache rebuild started in background"})


# ── Price history ─────────────────────────────────────────────────────────────
@app.route("/price-history")
def price_history():
    symbol  = request.args.get("symbol","").upper().strip()
    period  = request.args.get("period","1y")  # 1mo, 3mo, 1y, 5y
    if not symbol:
        return jsonify({"error":"symbol required"}),400
    try:
        import yfinance as yf
        import math
        period_map = {"1m":"1mo","3m":"3mo","1y":"1y","5y":"5y"}
        yf_period  = period_map.get(period,"1y")
        df = yf.download(f"{symbol}.NS", period=yf_period,
                         interval="1wk" if yf_period=="5y" else "1d",
                         auto_adjust=True, progress=False)
        if (df is None or len(df)==0) and yf_period=="5y":
            df = yf.download(f"{symbol}.NS", period="5y",
                             interval="1mo", auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return jsonify({"error":"no data"}),404
        if hasattr(df.columns,'levels'):
            df.columns = df.columns.get_level_values(0)
        closes = df['Close'].squeeze()
        dates  = [d.strftime("%Y-%m-%d") for d in closes.index]
        prices = [round(float(v),2) if not math.isnan(float(v)) else None
                  for v in closes.values]
        # Calculate % change from start
        start = next((p for p in prices if p), prices[0])
        pct_change = round((prices[-1]-start)/start*100,2) if start else 0
        return jsonify({
            "status":     "ok",
            "symbol":     symbol,
            "period":     period,
            "dates":      dates,
            "prices":     prices,
            "pct_change": pct_change,
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500


# ── Screener trend data ───────────────────────────────────────────────────────
@app.route("/trend-data")
def trend_data():
    symbol = request.args.get("symbol","").upper().strip()
    if not symbol:
        return jsonify({"error":"symbol required"}),400
    try:
        import math
        from screener_scraper import get_page, parse_table, parse_num

        SYMBOL_MAP = {'LTM':'LTIMINDTREE','M&M':'M&M','BAJAJ-AUTO':'BAJAJ-AUTO'}
        scr_sym = SYMBOL_MAP.get(symbol, symbol)

        soup = get_page(scr_sym)
        pl   = parse_table(soup,'profit-loss')
        bs   = parse_table(soup,'balance-sheet')
        rat  = parse_table(soup,'ratios')

        years = pl.get('_years',[])
        # Clean years — take last 10
        years = years[-10:] if len(years) > 10 else years

        def clean(arr):
            arr = arr[-10:] if len(arr) > 10 else arr
            return [round(float(v),1) if v is not None and not math.isnan(float(v if v else 0)) else None
                    for v in arr]

        sales   = clean(pl.get('Sales', pl.get('Revenue', pl.get('Interest Earned',[]))))
        profit  = clean(pl.get('Net Profit',[]))
        roce    = clean(rat.get('ROCE %', rat.get('ROE %',[])))
        debt    = clean(bs.get('Borrowings',[]))
        equity  = []
        eq_cap  = bs.get('Equity Capital',[])
        res     = bs.get('Reserves',[])
        if eq_cap and res:
            for i in range(max(len(eq_cap),len(res))):
                e = eq_cap[i] if i < len(eq_cap) else None
                r = res[i]    if i < len(res)    else None
                if e is not None and r is not None:
                    equity.append(round(float(e)+float(r),1))
                else:
                    equity.append(None)
            equity = clean(equity)

        # Promoter from shareholding table
        promoter  = []
        years_sh  = years
        sh_section = soup.find('section',{'id':'shareholding'})
        if sh_section:
            table = sh_section.find('table')
            if table:
                sh_years = []
                thead = table.find('thead')
                if thead:
                    sh_years = [th.get_text(strip=True) for th in thead.find_all('th')[1:]]
                tbody = table.find('tbody')
                if tbody:
                    for tr in tbody.find_all('tr'):
                        cells = tr.find_all('td')
                        if cells and 'Promoters' in cells[0].get_text():
                            promoter = clean([parse_num(td.get_text(strip=True)) for td in cells[1:]])
                            years_sh  = sh_years
                            break

        def fix_nan(obj):
            if isinstance(obj,dict):   return {k:fix_nan(v) for k,v in obj.items()}
            elif isinstance(obj,list): return [fix_nan(v) for v in obj]
            elif isinstance(obj,float) and math.isnan(obj): return None
            return obj

        return jsonify(fix_nan({
            "status":   "ok",
            "symbol":   symbol,
            "years":    years,
            "sales":    sales,
            "profit":   profit,
            "roce":     roce,
            "debt":     debt,
            "equity":   equity,
            "promoter": promoter,
            "promoter_years": years_sh,
        }))
    except Exception as e:
        return jsonify({"error":str(e)}),500


# ── Startup cache warming ─────────────────────────────────────────────────────
def warm_cache():
    """Pre-warm Nifty cache on server startup so first search is fast."""
    def _warm():
        print("  Warming Nifty cache on startup...")
        try:
            close = get_nifty_close()
            if close is not None:
                print(f"  Nifty cache ready — {len(close)} days")
            else:
                print("  Nifty cache warm failed")
        except Exception as e:
            print(f"  Nifty warm error: {e}")
    threading.Thread(target=_warm, daemon=True).start()


def warm_stock_features():
    """Pre-compute ML features for all stocks in background."""
    def _warm():
        time.sleep(30)  # Wait for Nifty cache to warm first
        print("  Pre-computing ML features for all stocks...")
        try:
            nifty = get_nifty_close()
            if nifty is None:
                return
            from scraper import NSE_STOCKS
            count = 0
            for sym in NSE_STOCKS:
                get_stock_features_cached(sym, nifty)
                count += 1
            print(f"  Pre-computed features for {count} stocks")
        except Exception as e:
            print(f"  Feature warm error: {e}")
    threading.Thread(target=_warm, daemon=True).start()


def warm_nse_session():
    """Pre-warm NSE session on startup."""
    def _warm():
        print("  Warming NSE session...")
        try:
            get_session()
            print("  NSE session ready")
        except Exception as e:
            print(f"  NSE session warm error: {e}")
    threading.Thread(target=_warm, daemon=True).start()


def warm_macro():
    """Pre-fetch macro sentiment on startup. Uses disk cache if fresh."""
    def _warm():
        try:
            from ml_screener import _cache
            cached = load_disk_cache('macro', max_age_hours=12)
            if cached:
                _cache['macro_news']['data'] = cached
                _cache['macro_news']['ts']   = time.time()
                print(f"  Macro cache loaded from disk — {len(cached)} topics")
                return
            time.sleep(60)
            print("  Warming macro sentiment cache (26 topics)...")
            from macro_sentiment import get_macro_sentiment
            macro_data = get_macro_sentiment()
            _cache['macro_news']['data'] = macro_data
            _cache['macro_news']['ts']   = time.time()
            save_disk_cache('macro', macro_data)
            print(f"  Macro cache ready and saved to disk — {len(macro_data)} topics")
        except Exception as e:
            print(f"  Macro warm error: {e}")
    threading.Thread(target=_warm, daemon=True).start()


warm_cache()
warm_stock_features()
warm_nse_session()
warm_macro()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nIndia Stock API running at http://localhost:5000")
    print("   Endpoints:")
    print("   GET /quote?symbol=RELIANCE")
    print("   GET /quotes?symbols=RELIANCE,TCS,INFY")
    print("   GET /watchlist")
    print("   GET /indices")
    print("   GET /health\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
