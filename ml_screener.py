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
import warnings
warnings.filterwarnings('ignore')

NSE_STOCKS = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN",
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
    "RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID", "NTPC",
    "ADANIPORTS", "ADANIENT", "TATAPOWER",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
    "LT", "ULTRACEMCO", "GRASIM",
    "BHARTIARTL", "ASIANPAINT", "TITAN",
]

# Cache predictions for 1 hour
_cache = {'predictions': None, 'ts': 0}
CACHE_TTL = 3600


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


def run_ml_screen(top_n=10):
    import time
    now = time.time()

    if _cache['predictions'] and (now - _cache['ts']) < CACHE_TTL:
        return _cache['predictions']

    model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
    if not os.path.exists(model_path):
        return {'error': 'Model not trained yet. Run ml_train.py first.'}

    saved    = joblib.load(model_path)
    model    = saved['model']
    features = saved['features']
    accuracy = saved['accuracy']

    nf = yf.download("^NSEI", period="2y", interval="1d",
                     auto_adjust=True, progress=False)
    if hasattr(nf.columns, 'levels'):
        nf.columns = nf.columns.get_level_values(0)
    nifty = pd.Series(nf['Close'].squeeze().values,
                      index=nf.index, dtype=float)

    results = []
    for sym in NSE_STOCKS:
        f = get_features(sym, nifty)
        if not f:
            continue
        X    = pd.DataFrame([{k: f[k] for k in features}])
        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        results.append({
            'symbol':       sym,
            'price':        f['price'],
            'ml_score':     round(prob * 100, 1),
            'prediction':   'OUTPERFORM' if pred == 1 else 'UNDERPERFORM',
            'rsi':          round(f['rsi'], 1),
            'pos52_pct':    round(f['pos52'] * 100, 1),
            'ret_1m_pct':   round(f['ret_1m'] * 100, 1),
            'ret_3m_pct':   round(f['ret_3m'] * 100, 1),
            'rs_3m_pct':    round(f['rs_3m'] * 100, 1),
            'golden_cross': bool(f['golden_cross']),
            'vol_3m_pct':   round(f['vol_3m'] * 100, 1),
        })

    results.sort(key=lambda x: x['ml_score'], reverse=True)

    output = {
        'generated_at':    datetime.now().isoformat(),
        'model_accuracy':  round(accuracy * 100, 1),
        'stocks_screened': len(results),
        'top10':           results[:top_n],
        'bottom5':         results[-5:],
        'all':             results,
    }

    _cache['predictions'] = output
    _cache['ts'] = now
    return output
