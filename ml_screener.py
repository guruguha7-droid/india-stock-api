"""
ML Screener — API integration module
Called by api_server.py for /ml-screen endpoint
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from news_sentiment import get_sentiment_score
import warnings
warnings.filterwarnings('ignore')

NSE_STOCKS = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "TCS", "INFY",
    "WIPRO", "HCLTECH", "TECHM", "LTM", "RELIANCE",
    "ONGC", "BPCL", "IOC", "POWERGRID", "NTPC",
    "ADANIPORTS", "ADANIENT", "TATAPOWER", "HINDUNILVR", "ITC",
    "NESTLEIND", "BRITANNIA", "TATACONSUM", "MARUTI", "M&M",
    "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "SUNPHARMA", "DRREDDY",
    "CIPLA", "DIVISLAB", "APOLLOHOSP", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "COALINDIA", "VEDL", "LT", "ULTRACEMCO",
    "GRASIM", "BHARTIARTL", "ASIANPAINT", "TITAN", "BANKBARODA",
    "PNB", "CANBK", "MUTHOOTFIN", "CHOLAFIN", "MANAPPURAM",
    "MARICO", "DABUR", "COLPAL", "GODREJCP", "EMAMILTD",
    "TORNTPHARM", "LUPIN", "AUROPHARMA", "ALKEM", "PERSISTENT",
    "MPHASIS", "COFORGE", "KPITTECH", "TVSMOTOR", "MOTHERSON",
    "BALKRISIND", "APOLLOTYRE", "SIEMENS", "HAVELLS", "ABB",
    "CUMMINSIND", "DLF", "OBEROIRLTY", "RAMCOCEM", "FEDERALBNK",
    "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "RBLBANK", "INDIANB",
    "MAHABANK", "HDFCAMC", "ICICIGI", "SBICARD",
    "SUNDARMFIN", "TATAELXSI", "LTTS", "HAPPSTMNDS", "ZENSARTECH",
    "MANKIND", "ABBOTINDIA", "NATCOPHARM", "GRANULES", "GLENMARK",
    "IPCALAB", "MAXHEALTH", "RADICO", "UBL", "VBL",
    "BIKAJI", "ZYDUSWELL", "ASHOKLEY", "BOSCHLTD", "TIINDIA",
    "ENDURANCE", "SUNDRMFAST", "SCHAEFFLER", "ADANIGREEN", "ADANIPOWER",
    "TORNTPOWER", "CESC", "NHPC", "SJVN", "HINDPETRO",
    "GAIL", "AMBUJACEM", "ACC", "JKCEMENT",
    "IRB", "KNRCON", "NMDC", "SAIL", "NATIONALUM",
    "MOIL", "WELCORP", "GODREJPROP", "BRIGADE", "SOBHA",
    "PRESTIGE", "PHOENIXLTD", "HAL", "BEL", "BEML",
    "MAZDOCK",
]

# Separate caches for different data types
_cache = {
    'ml_features':  {'data': None, 'ts': 0},  # ML + Screener + yfinance
    'company_news': {'data': None, 'ts': 0},  # Company sentiment
    'macro_news':   {'data': None, 'ts': 0},  # Macro sentiment
    'combined':     {'data': None, 'ts': 0},  # Final combined output
}

ML_TTL       = 21600  # 6 hours  — ML features don't change fast
NEWS_TTL     = 1800   # 30 mins  — Company news changes fast
MACRO_TTL    = 7200   # 2 hours  — Macro news changes moderately
COMBINED_TTL = 1800   # 30 mins  — Combined refreshes when news refreshes


def load_screener_fundamentals() -> dict:
    """Load pre-scraped Screener.in fundamental data."""
    path = os.path.join(os.path.dirname(__file__), 'screener_fundamentals.csv')
    if not os.path.exists(path):
        print("  Warning: screener_fundamentals.csv not found — run screener_scraper.py first")
        return {}
    try:
        df = pd.read_csv(path)
        df = df[df['status'] == 'ok']
        return df.set_index('symbol').to_dict(orient='index')
    except Exception as e:
        print(f"  Warning: Could not load screener data — {e}")
        return {}


def get_features(sym, nifty_close):
    try:
        df = yf.download(f"{sym}.NS", period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        if len(df) < 200:
            return None
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        close = pd.Series(df['Close'].squeeze().values,
                          index=df.index, dtype=float)
        common = close.index.intersection(nifty_close.index)
        if len(common) < 200:
            return None
        sw = close.loc[common].values.astype(float)
        nw = nifty_close.loc[common].values.astype(float)
        cp = float(sw[-1])
        if cp <= 0 or np.isnan(cp):
            return None

        def sr(arr, b):
            return float(arr[-1]/arr[-b]-1) if len(arr)>b and arr[-b]>0 else 0.0

        ret_1m = sr(sw,22); ret_3m = sr(sw,63)
        ret_6m = sr(sw,126); ret_1y = sr(sw,min(200,len(sw)-1))
        rs_1m  = ret_1m - sr(nw,22)
        rs_3m  = ret_3m - sr(nw,63)
        ma50   = float(np.mean(sw[-50:]))
        ma200  = float(np.mean(sw[-200:]))
        dr     = np.diff(sw)/sw[:-1]
        dr     = dr[~np.isnan(dr)]
        vol_1m = float(np.std(dr[-22:])*np.sqrt(252)) if len(dr)>=22 else 0.3
        vol_3m = float(np.std(dr[-63:])*np.sqrt(252)) if len(dr)>=63 else 0.3
        h52 = float(np.max(sw[-252:])) if len(sw)>=252 else float(np.max(sw))
        l52 = float(np.min(sw[-252:])) if len(sw)>=252 else float(np.min(sw))
        rng = h52 - l52
        d   = np.diff(sw[-16:])
        g   = d[d>0].mean() if len(d[d>0])>0 else 0.001
        ls  = abs(d[d<0].mean()) if len(d[d<0])>0 else 0.001

        return {
            'symbol': sym, 'price': round(cp, 2),
            'ret_1m': ret_1m, 'ret_3m': ret_3m,
            'ret_6m': ret_6m, 'ret_1y': ret_1y,
            'rs_1m': rs_1m, 'rs_3m': rs_3m,
            'price_to_ma50':  cp/ma50-1  if ma50>0  else 0,
            'price_to_ma200': cp/ma200-1 if ma200>0 else 0,
            'golden_cross': 1 if ma50>ma200 else 0,
            'vol_1m': vol_1m, 'vol_3m': vol_3m,
            'pos52': float((cp-l52)/rng) if rng>0 else 0.5,
            'rsi':   float(100-100/(1+g/ls)),
            'vol_trend': float(vol_1m/vol_3m) if vol_3m>0 else 1.0,
        }
    except Exception:
        return None


def get_ml_features(nifty_close):
    """Compute ML features for all stocks. Cached 6 hours."""
    import time
    now   = time.time()
    cache = _cache['ml_features']

    if cache['data'] and (now - cache['ts']) < ML_TTL:
        print("  ML features: using cache")
        return cache['data']

    print("  ML features: computing fresh...")
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
    if not os.path.exists(model_path):
        return None

    saved    = joblib.load(model_path)
    model    = saved['model']
    features = saved['features']
    accuracy = saved['accuracy']

    all_features = []
    for sym in NSE_STOCKS:
        f = get_features(sym, nifty_close)
        if f:
            all_features.append(f)

    # Screener fundamentals
    screener_data = load_screener_fundamentals()

    # yfinance fundamentals
    yfin_data = {}
    for sym in NSE_STOCKS:
        try:
            info = yf.Ticker(f"{sym}.NS").info
            yfin_data[sym] = {
                'pe': float(info.get('trailingPE')    or 20),
                'pm': float(info.get('profitMargins') or 0.10),
                'rg': float(info.get('revenueGrowth') or 0.10),
                'eg': float(info.get('earningsGrowth')or 0.10),
            }
        except Exception:
            yfin_data[sym] = {'pe': 20, 'pm': 0.10, 'rg': 0.10, 'eg': 0.10}

    result = {
        'model':         model,
        'features':      features,
        'accuracy':      accuracy,
        'all_features':  all_features,
        'screener_data': screener_data,
        'yfin_data':     yfin_data,
    }
    cache['data'] = result
    cache['ts']   = now
    return result


def get_company_news(all_features):
    """Fetch company news sentiment. Cached 30 minutes."""
    import time
    now   = time.time()
    cache = _cache['company_news']

    if cache['data'] and (now - cache['ts']) < NEWS_TTL:
        print("  Company news: using cache (refreshes every 30 mins)")
        return cache['data']

    print("  Company news: fetching fresh...")
    sentiment_data = {}
    for f in all_features:
        sym = f['symbol']
        sentiment_data[sym] = get_sentiment_score(sym)
        time.sleep(0.5)

    cache['data'] = sentiment_data
    cache['ts']   = now
    return sentiment_data


def get_macro_news():
    """Fetch macro sentiment. Cached 2 hours."""
    import time
    now   = time.time()
    cache = _cache['macro_news']

    if cache['data'] and (now - cache['ts']) < MACRO_TTL:
        print("  Macro news: using cache (refreshes every 2 hrs)")
        return cache['data']

    print("  Macro news: fetching fresh...")
    from macro_sentiment import get_macro_sentiment
    macro_data = get_macro_sentiment()

    cache['data'] = macro_data
    cache['ts']   = now
    return macro_data


def run_ml_screen(top_n=10):
    """
    Main screening function with separate caches:
    - ML features:    6 hours
    - Company news:   30 minutes
    - Macro news:     2 hours
    - Combined score: recalculated when any cache is fresh
    """
    import time, math
    now = time.time()

    # Invalidate combined if news cache is stale
    combined_cache = _cache['combined']
    news_cache     = _cache['company_news']
    if news_cache['ts'] > combined_cache['ts']:
        combined_cache['data'] = None

    if combined_cache['data'] and (now - combined_cache['ts']) < COMBINED_TTL:
        print("  Combined: using cache")
        return combined_cache['data']

    # ── Download Nifty benchmark ──────────────────────────────────────
    print("  Downloading Nifty benchmark...")
    nifty_df = None
    for _ in range(3):
        try:
            nifty_df = yf.download("^NSEI", period="2y", interval="1d",
                                   auto_adjust=True, progress=False)
            if len(nifty_df) > 100:
                break
            time.sleep(15)
        except Exception:
            time.sleep(15)

    if nifty_df is None or len(nifty_df) < 100:
        return {'error': 'Could not download Nifty data'}

    if hasattr(nifty_df.columns, 'levels'):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_close = pd.Series(nifty_df['Close'].squeeze().values,
                            index=nifty_df.index, dtype=float)

    # ── Get all data layers ───────────────────────────────────────────
    ml_data = get_ml_features(nifty_close)
    if not ml_data:
        return {'error': 'Model not trained yet. Run ml_train.py first.'}

    company_news = get_company_news(ml_data['all_features'])
    macro_data   = get_macro_news()

    model         = ml_data['model']
    features      = ml_data['features']
    accuracy      = ml_data['accuracy']
    all_features  = ml_data['all_features']
    screener_data = ml_data['screener_data']
    yfin_data     = ml_data['yfin_data']

    # ── Score all stocks ──────────────────────────────────────────────
    from macro_sentiment import apply_macro_to_stock
    results = []

    for f in all_features:
        sym  = f['symbol']
        X    = pd.DataFrame([{k: f[k] for k in features}])
        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        ml_raw = round(prob * 100, 1)

        # Screener fundamentals
        _CSV_MAP = {'VBL': 'VBLLTD', 'LTM': 'LTIMINDTREE'}
        sc             = screener_data.get(_CSV_MAP.get(sym, sym), {})
        screener_score = float(sc.get('investment_score', 50) or 50)
        screener_grade = sc.get('investment_grade', 'C') or 'C'
        roce           = float(sc.get('roce_latest_pct', 10) or 10)
        sales_cagr_5y  = float(sc.get('sales_cagr_5y', 10) or 10)
        profit_cagr_5y = float(sc.get('profit_cagr_5y', 10) or 10)
        promoter_pct   = float(sc.get('promoter_pct', 30) or 30)
        fcf_positive   = bool(sc.get('fcf_positive_3y', False))
        debt_reducing  = bool(sc.get('debt_reducing', False))

        # yfinance fundamentals
        yf_d = yfin_data.get(sym, {})
        pe = float(yf_d.get('pe', 20) or 20)
        pm = float(yf_d.get('pm', 0.10) or 0.10)
        rg = float(yf_d.get('rg', 0.10) or 0.10)
        eg = float(yf_d.get('eg', 0.10) or 0.10)

        yfin_score = 50
        if pe < 12:     yfin_score += 20
        elif pe < 18:   yfin_score += 12
        elif pe < 25:   yfin_score += 5
        elif pe > 40:   yfin_score -= 15
        if pm > 0.20:   yfin_score += 10
        elif pm > 0.12: yfin_score += 5
        elif pm < 0:    yfin_score -= 15
        if rg > 0.20:   yfin_score += 10
        elif rg > 0.10: yfin_score += 5
        elif rg < -0.05:yfin_score -= 8
        if eg > 0.20:   yfin_score += 8
        elif eg > 0.05: yfin_score += 4
        elif eg < -0.10:yfin_score -= 8
        yfin_score = max(0, min(100, yfin_score))

        # Company news sentiment
        sent       = company_news.get(sym, {})
        sent_raw   = float(sent.get('sentiment_score', 0) or 0)
        sent_label = sent.get('sentiment_label', 'neutral')
        sent_score = round(50 + sent_raw * 0.5, 1)
        sent_score = max(0, min(100, sent_score))

        # Macro sentiment
        macro       = apply_macro_to_stock(sym, macro_data)
        macro_raw   = float(macro.get('macro_score', 0) or 0)
        macro_label = macro.get('macro_label', 'neutral')
        macro_score = round(50 + macro_raw * 0.5, 1)
        macro_score = max(0, min(100, macro_score))

        # 5-layer combined score
        combined = round(
            ml_raw         * 0.35 +
            screener_score * 0.20 +
            yfin_score     * 0.15 +
            sent_score     * 0.15 +
            macro_score    * 0.15,
            1
        )

        if combined >= 80:   inv_grade = 'A+'
        elif combined >= 70: inv_grade = 'A'
        elif combined >= 60: inv_grade = 'B'
        elif combined >= 50: inv_grade = 'C'
        else:                inv_grade = 'D'

        results.append({
            'symbol':          sym,
            'price':           f['price'],
            'ml_score':        ml_raw,
            'screener_score':  round(screener_score, 1),
            'screener_grade':  screener_grade,
            'yfin_score':      round(yfin_score, 1),
            'sentiment_score': sent_raw,
            'sentiment_label': sent_label,
            'macro_score':     macro_raw,
            'macro_label':     macro_label,
            'combined_score':  combined,
            'prediction':      'OUTPERFORM' if pred == 1 else 'UNDERPERFORM',
            'inv_grade':       inv_grade,
            'roce':            roce,
            'sales_cagr_5y':   sales_cagr_5y,
            'profit_cagr_5y':  profit_cagr_5y,
            'promoter_pct':    promoter_pct,
            'fcf_positive':    fcf_positive,
            'debt_reducing':   debt_reducing,
            'pe_ratio':        round(pe, 1),
            'profit_margin':   round(pm * 100, 1),
            'revenue_growth':  round(rg * 100, 1),
            'rsi':             round(f['rsi'], 1),
            'pos52_pct':       round(f['pos52'] * 100, 1),
            'ret_1m_pct':      round(f['ret_1m'] * 100, 1),
            'ret_3m_pct':      round(f['ret_3m'] * 100, 1),
            'rs_3m_pct':       round(f['rs_3m'] * 100, 1),
            'golden_cross':    bool(f['golden_cross']),
            'vol_3m_pct':      round(f['vol_3m'] * 100, 1),
            'top_headlines':   sent.get('top_headlines', []),
            'macro_topics':    macro.get('topics', []),
        })

    results.sort(key=lambda x: x['combined_score'], reverse=True)

    def fix_nan(obj):
        if isinstance(obj, dict):
            return {k: fix_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_nan(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    output = fix_nan({
        'generated_at':    datetime.now().isoformat(),
        'model_accuracy':  round(accuracy * 100, 1),
        'stocks_screened': len(results),
        'cache_info': {
            'ml_features_age_mins':  round((now - _cache['ml_features']['ts']) / 60, 1),
            'company_news_age_mins': round((now - _cache['company_news']['ts']) / 60, 1),
            'macro_news_age_mins':   round((now - _cache['macro_news']['ts']) / 60, 1),
        },
        'top10':   results[:top_n],
        'bottom5': results[-5:],
        'all':     results,
    })

    combined_cache['data'] = output
    combined_cache['ts']   = now
    return output
