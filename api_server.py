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
        try:
            import joblib
            import pandas as pd
            import yfinance as _yf
            import numpy as np

            saved    = joblib.load(os.path.join(os.path.dirname(__file__), 'ml_model.pkl'))
            model    = saved['model']
            features = saved['features']
            accuracy = saved['accuracy']

            # Patch sklearn version mismatch — add _fill_dtype if missing from SimpleImputer
            try:
                steps = model.steps if hasattr(model, 'steps') else []
                for _, step in steps:
                    if hasattr(step, 'statistics_') and not hasattr(step, '_fill_dtype'):
                        step._fill_dtype = step.statistics_.dtype
            except Exception:
                pass

            # Always download fresh — bypass all caches for single-stock calls
            stock_df = _yf.download(f"{symbol}.NS", period="2y", interval="1d",
                                    auto_adjust=True, progress=False)
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

            # Align with Nifty for relative-strength features; graceful fallback if Nifty fails
            nw = None
            if nifty_df is not None and len(nifty_df) >= 100:
                nw_s   = pd.Series(nifty_df['Close'].squeeze().values,
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

            d_rsi = np.diff(sw[-16:]) if len(sw) >= 16 else np.array([0.001, -0.001])
            g     = d_rsi[d_rsi > 0].mean() if len(d_rsi[d_rsi > 0]) > 0 else 0.001
            ls    = abs(d_rsi[d_rsi < 0].mean()) if len(d_rsi[d_rsi < 0]) > 0 else 0.001

            f = {
                'ret_1m': ret_1m,           'ret_3m': ret_3m,
                'ret_6m': ret_6m,           'ret_1y': ret_1y,
                'rs_1m':  rs_1m,            'rs_3m':  rs_3m,
                'price_to_ma50':  cp / ma50  - 1 if ma50  > 0 else 0,
                'price_to_ma200': cp / ma200 - 1 if ma200 > 0 else 0,
                'golden_cross':   1 if ma50 > ma200 else 0,
                'vol_1m': vol_1m,           'vol_3m': vol_3m,
                'pos52':  float((cp - l52) / rng) if rng > 0 else 0.5,
                'rsi':    float(100 - 100 / (1 + g / ls)),
                'vol_trend': float(vol_1m / vol_3m) if vol_3m > 0 else 1.0,
                # Fundamental defaults — overwritten from CSV below
                'roce_latest_pct': 12.0, 'opm_latest_pct':  12.0,
                'sales_cagr_5y':   10.0, 'profit_cagr_5y':   8.0,
                'eps_cagr_5y':      8.0, 'sales_growth_1y':  8.0,
                'profit_growth_1y': 8.0, 'opm_trend_5y':     0.0,
                'roce_trend_5y':    0.0, 'promoter_pct':    45.0,
                'fii_pct':         15.0, 'fcf_positive_3y':  0.5,
                'debt_reducing':    0.5, 'screener_de':     50.0,
            }

            # Inject CSV fundamentals — overwrites defaults above
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
                pass  # defaults already set above

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
                    "roce":             r.get('roce_latest_pct'),
                    "sales_cagr_5y":    r.get('sales_cagr_5y'),
                    "profit_cagr_5y":   r.get('profit_cagr_5y'),
                    "eps_cagr_5y":      r.get('eps_cagr_5y'),
                    "promoter_pct":     r.get('promoter_pct'),
                    "fcf_positive_3y":  r.get('fcf_positive_3y'),
                    "debt_reducing":    r.get('debt_reducing'),
                    "investment_score": r.get('investment_score'),
                    "investment_grade": r.get('investment_grade'),
                    "opm_latest_pct":   r.get('opm_latest_pct'),
                    "roce_avg_5y":      r.get('roce_avg_5y'),
                }
            else:
                result["fundamentals"] = {}
        except Exception:
            result["fundamentals"] = {}

    def fetch_valuation():
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
                    one_yr = divs[divs.index >= (pd.Timestamp.now() - pd.DateOffset(years=1))]
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
            result["sentiment"] = get_sentiment_score(symbol)
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
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=40)

    # ── Combined score ────────────────────────────────────────────────
    try:
        ml_raw  = result.get("ml", {}).get("ml_score", 50) or 50
        scr_raw = result.get("fundamentals", {}).get("investment_score", 50) or 50

        pe = result.get("valuation", {}).get("pe_ratio") or 20
        pm = result.get("valuation", {}).get("profit_margin") or 0.10
        rg = result.get("valuation", {}).get("revenue_growth") or 0.10

        yfin_score = 50
        if pe < 12:      yfin_score += 20
        elif pe < 18:    yfin_score += 12
        elif pe < 25:    yfin_score += 5
        elif pe > 40:    yfin_score -= 15
        if pm > 0.20:    yfin_score += 10
        elif pm > 0.12:  yfin_score += 5
        elif pm < 0:     yfin_score -= 15
        if rg > 0.15:    yfin_score += 10
        elif rg > 0.08:  yfin_score += 5
        elif rg < -0.05: yfin_score -= 8
        yfin_score = max(0, min(100, yfin_score))

        sent_raw   = result.get("sentiment", {}).get("sentiment_score", 0) or 0
        sent_score = max(0, min(100, 50 + sent_raw * 0.5))

        macro_raw   = result.get("macro", {}).get("macro_score", 0) or 0
        macro_score = max(0, min(100, 50 + macro_raw * 0.5))

        combined = round(
            ml_raw      * 0.35 +
            scr_raw     * 0.20 +
            yfin_score  * 0.15 +
            sent_score  * 0.15 +
            macro_score * 0.15, 1)

        grade = 'A+' if combined >= 80 else 'A' if combined >= 70 else 'B' if combined >= 60 else 'C' if combined >= 50 else 'D'

        # ── Base verdict ──────────────────────────────────────────────
        if combined >= 65:   verdict, verdict_color = 'BUY',  'green'
        elif combined >= 50: verdict, verdict_color = 'HOLD', 'gold'
        else:                verdict, verdict_color = 'SELL', 'red'

        # ── Contrarian override ───────────────────────────────────────
        rsi_val  = float(result.get('ml', {}).get('rsi') or 50)
        pos52    = float(result.get('ml', {}).get('pos52_pct') or 50)
        ret_1m   = float(result.get('ml', {}).get('ret_1m_pct') or 0)

        contrarian = (
            sent_raw  < -15  and
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

        de  = float(result.get('valuation', {}).get('debt_to_equity') or 50)
        vol = abs(float(result.get('ml', {}).get('ret_1m_pct') or 0))

        if is_large and de < 60 and vol < 10:  risk, risk_color = 'Low',    'green'
        elif is_large or de < 100:             risk, risk_color = 'Medium', 'gold'
        else:                                  risk, risk_color = 'High',   'red'

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

        reason   = reasons[0].capitalize() + (', ' + reasons[1] if len(reasons) > 1 else '')
        score_10 = round(combined / 10, 1)

        result["combined"] = {
            "score":          combined,
            "grade":          grade,
            "yfin_score":     round(yfin_score, 1),
            "sent_score":     round(sent_score, 1),
            "macro_score":    round(macro_score, 1),
            "screener_score": round(scr_raw, 1),
            "verdict":        verdict,
            "verdict_color":  verdict_color,
            "risk":           risk,
            "risk_color":     risk_color,
            "reason":         reason,
            "score_10":       score_10,
        }
    except Exception:
        result["combined"] = {"score": 50, "grade": "C"}

    # ── Forecast ──────────────────────────────────────────────────────
    try:
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
        sales_cagr  = float(fund.get("sales_cagr_5y")  or 8)
        profit_cagr = float(fund.get("profit_cagr_5y") or 8)
        eps_cagr    = float(fund.get("eps_cagr_5y") or profit_cagr)
        roce_latest = float(fund.get("roce") or fund.get("roce_latest_pct") or 10)
        roce_avg    = float(fund.get("roce_avg_5y") or roce_latest)
        ocf         = fund.get("ocf_latest_cr")
        fcf_ok      = fund.get("fcf_positive_3y")
        debt_red    = fund.get("debt_reducing")
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
        if fcf_ok is True:   capex_mult += 0.05
        if ocf and float(ocf) > 0: capex_mult += 0.03
        capex_mult = clamp(capex_mult, 0.95, 1.08)

        # 5. Debt trend multiplier — reducing debt = lower risk
        debt_mult = 1.03 if debt_red is True else 0.97 if debt_red is False else 1.0

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

        adj_eps_cagr_1y    = eps_cagr    * base_mult * news_mult_1y
        adj_eps_cagr_3y    = eps_cagr    * base_mult * news_mult_3y * 0.92
        adj_eps_cagr_5y    = eps_cagr    * base_mult * news_mult_5y * 0.85
        adj_sales_cagr_1y  = sales_cagr  * base_mult * news_mult_1y
        adj_sales_cagr_3y  = sales_cagr  * base_mult * news_mult_3y * 0.92
        adj_sales_cagr_5y  = sales_cagr  * base_mult * news_mult_5y * 0.85
        adj_profit_cagr_1y = profit_cagr * base_mult * news_mult_1y
        adj_profit_cagr_3y = profit_cagr * base_mult * news_mult_3y * 0.92
        adj_profit_cagr_5y = profit_cagr * base_mult * news_mult_5y * 0.85

        # ══════════════════════════════════════════════════════════════
        # PRICE TARGET
        # ══════════════════════════════════════════════════════════════
        def price_target(years, eps_cagr_adj, news_m, mom_m=1.0):
            if not eps or not pe: return None
            fwd_eps = float(eps) * ((1 + eps_cagr_adj / 100) ** years)
            raw     = fwd_eps * pe * news_m * mom_m
            return round(raw, 0)

        pt_1y = price_target(1, adj_eps_cagr_1y, news_mult_1y, momentum_mult)
        pt_3y = price_target(3, adj_eps_cagr_3y, news_mult_3y)
        pt_5y = price_target(5, adj_eps_cagr_5y, news_mult_5y)

        # ══════════════════════════════════════════════════════════════
        # OUTPERFORM CONFIDENCE
        # ══════════════════════════════════════════════════════════════
        scr_score = float(fund.get("investment_score") or 50)
        def outperform_prob(years):
            base  = ml_score * 0.50 + scr_score * 0.30 + (50 + macro_score * 0.20) * 0.20
            decay = 0.85 ** years
            return round(clamp(50 + (base - 50) * decay, 20, 85), 1)

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
                "revenue_growth_pct": round(adj_sales_cagr_1y, 1),
                "profit_growth_pct":  round(adj_profit_cagr_1y, 1),
                "outperform_prob":    outperform_prob(1),
            },
            "3y": {
                "price_target":       pt_3y,
                "revenue_growth_pct": round(adj_sales_cagr_3y, 1),
                "profit_growth_pct":  round(adj_profit_cagr_3y, 1),
                "outperform_prob":    outperform_prob(3),
            },
            "5y": {
                "price_target":       pt_5y,
                "revenue_growth_pct": round(adj_sales_cagr_5y, 1),
                "profit_growth_pct":  round(adj_profit_cagr_5y, 1),
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
            q = nse_quote(sym)
            fund = {}
            try:
                import pandas as pd
                path = os.path.join(os.path.dirname(__file__),'screener_fundamentals.csv')
                sdf  = pd.read_csv(path)
                SYMBOL_CSV_MAP = {'LTM':'LTIM'}
                csv_sym = SYMBOL_CSV_MAP.get(sym, sym)
                row = sdf[sdf['symbol']==csv_sym]
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    fund = {
                        'roce':            r.get('roce_latest_pct'),
                        'sales_cagr_5y':   r.get('sales_cagr_5y'),
                        'profit_cagr_5y':  r.get('profit_cagr_5y'),
                        'investment_score':r.get('investment_score'),
                        'investment_grade':r.get('investment_grade'),
                    }
            except Exception:
                pass

            val = {}
            try:
                import yfinance as yf
                info = yf.Ticker(f"{sym}.NS").info
                val = {
                    'pe_ratio':  info.get('trailingPE'),
                    'eps':       info.get('trailingEps'),
                    'div_yield': _safe_div_yield(info.get('dividendYield'),info.get('dividendRate'),info.get('currentPrice')),
                    'ret_1y':    info.get('52WeekChange'),
                }
            except Exception:
                pass

            ml = {}
            try:
                import joblib
                import pandas as _pd
                saved = joblib.load(os.path.join(os.path.dirname(__file__),'ml_model.pkl'))
                nifty = get_nifty_close()
                if nifty is not None:
                    f = get_stock_features_cached(sym, nifty)
                    if f:
                        X    = _pd.DataFrame([{k:f[k] for k in saved['features']}])
                        prob = float(saved['model'].predict_proba(X)[0][1])
                        ml   = {
                            'ml_score':   round(prob*100,1),
                            'ret_1m_pct': round(f['ret_1m']*100,1),
                            'ret_3m_pct': round(f['ret_3m']*100,1),
                        }
            except Exception:
                pass

            pt_1y = None
            try:
                eps     = val.get('eps')
                pe      = float(val.get('pe_ratio') or 20)
                ep_cagr = float(fund.get('profit_cagr_5y') or 8)
                if eps:
                    pt_1y = round(float(eps)*((1+ep_cagr/100)**1)*pe,0)
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
                    'ret_1y':          round(float(val.get('ret_1y') or 0)*100,1) if val.get('ret_1y') else None,
                    'pe_ratio':        val.get('pe_ratio'),
                    'eps':             val.get('eps'),
                    'div_yield':       round(float(val.get('div_yield') or 0),2) if val.get('div_yield') else None,
                    'roce':            fund.get('roce'),
                    'sales_cagr_5y':   fund.get('sales_cagr_5y'),
                    'profit_cagr_5y':  fund.get('profit_cagr_5y'),
                    'ml_score':        ml.get('ml_score'),
                    'screener_score':  fund.get('investment_score'),
                    'screener_grade':  fund.get('investment_grade'),
                    'combined_grade':  None,  # TODO: compute full combined
                    'price_target_1y': pt_1y,
                }
        except Exception as e:
            with lock:
                results[sym] = {'symbol':sym,'error':str(e)}

    threads = [threading.Thread(target=fetch_one,args=(s,)) for s in symbols]
    for t in threads: t.start()
    for t in threads: t.join(timeout=25)

    def fix_nan(obj):
        if isinstance(obj,dict):   return {k:fix_nan(v) for k,v in obj.items()}
        elif isinstance(obj,list): return [fix_nan(v) for v in obj]
        elif isinstance(obj,float) and math.isnan(obj): return None
        return obj

    ordered = [fix_nan(results.get(s,{'symbol':s,'error':'timeout'})) for s in symbols]
    return jsonify({"status":"ok","count":len(ordered),"data":ordered})


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
    """Pre-fetch macro sentiment on startup. Fires 60s after boot."""
    def _warm():
        time.sleep(60)  # Let Nifty + NSE session warm first
        print("  Warming macro sentiment cache (26 topics)...")
        try:
            from macro_sentiment import get_macro_sentiment
            from ml_screener import _cache
            macro_data = get_macro_sentiment()
            _cache['macro_news']['data'] = macro_data
            _cache['macro_news']['ts']   = time.time()
            print(f"  Macro cache ready — {len(macro_data)} topics fetched")
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
