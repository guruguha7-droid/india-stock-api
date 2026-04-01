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
            saved    = joblib.load(os.path.join(os.path.dirname(__file__), 'ml_model.pkl'))
            model    = saved['model']
            features = saved['features']
            accuracy = saved['accuracy']

            nifty_close = get_nifty_close()
            if nifty_close is None:
                result["ml"] = {"error": "Nifty data unavailable"}
                return

            f = get_stock_features_cached(symbol, nifty_close)
            if f:
                X    = pd.DataFrame([{k: f[k] for k in features}])
                prob = float(model.predict_proba(X)[0][1])
                pred = int(model.predict(X)[0])
                result["ml"] = {
                    "ml_score":     round(prob * 100, 1),
                    "prediction":   "OUTPERFORM" if pred == 1 else "UNDERPERFORM",
                    "accuracy":     round(accuracy * 100, 1),
                    "rsi":          round(f['rsi'], 1),
                    "pos52_pct":    round(f['pos52'] * 100, 1),
                    "ret_1m_pct":   round(f['ret_1m'] * 100, 1),
                    "ret_3m_pct":   round(f['ret_3m'] * 100, 1),
                    "golden_cross": bool(f['golden_cross']),
                }
            else:
                result["ml"] = {"error": "Could not compute features"}
        except Exception as e:
            result["ml"] = {"error": str(e)}

    def fetch_fundamentals():
        try:
            import pandas as pd
            sdf = pd.read_csv(
                os.path.join(os.path.dirname(__file__), 'screener_fundamentals.csv'))
            row = sdf[sdf['symbol'] == symbol]
            if not row.empty:
                r = row.iloc[0].to_dict()
                result["fundamentals"] = {
                    "roce":             r.get('roce_latest_pct'),
                    "sales_cagr_5y":    r.get('sales_cagr_5y'),
                    "profit_cagr_5y":   r.get('profit_cagr_5y'),
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
        try:
            import yfinance as yf
            info = yf.Ticker(f"{symbol}.NS").info
            result["valuation"] = {
                "pe_ratio":        info.get('trailingPE'),
                "pb_ratio":        info.get('priceToBook'),
                "profit_margin":   info.get('profitMargins'),
                "revenue_growth":  info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth'),
                "debt_to_equity":  info.get('debtToEquity'),
                "dividend_yield":  info.get('dividendYield'),
                "eps":             info.get('trailingEps'),
            }
        except Exception:
            result["valuation"] = {}

    def fetch_sentiment():
        try:
            result["sentiment"] = get_sentiment_score(symbol)
        except Exception:
            result["sentiment"] = {"sentiment_score": 0, "sentiment_label": "neutral"}

    def fetch_macro():
        try:
            from ml_screener import _cache
            macro_cache = _cache.get('macro_news', {})
            if macro_cache.get('data') and macro_cache.get('ts', 0) > 0:
                macro_data = macro_cache['data']
            else:
                macro_data = {}
            result["macro"] = apply_macro_to_stock(symbol, macro_data)
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
        t.join(timeout=15)

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

        result["combined"] = {
            "score":       combined,
            "grade":       grade,
            "yfin_score":  round(yfin_score, 1),
            "sent_score":  round(sent_score, 1),
            "macro_score": round(macro_score, 1),
        }
    except Exception:
        result["combined"] = {"score": 50, "grade": "C"}

    def fix_nan(obj):
        if isinstance(obj, dict):
            return {k: fix_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_nan(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return jsonify(fix_nan(result))


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


warm_cache()
warm_stock_features()
warm_nse_session()


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
