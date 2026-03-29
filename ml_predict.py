"""
ML Stock Screener — Prediction Script
=======================================
Uses the trained model to predict which NSE stocks
are most likely to outperform Nifty in the next 3 months.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
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


def get_current_features(sym: str, nifty_close: pd.Series) -> dict:
    """
    Download recent price data for a stock and compute
    the same features the model was trained on.
    """
    try:
        df = yf.download(f"{sym}.NS", period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        if len(df) < 200:
            return None

        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        close = df['Close'].squeeze()
        close = pd.Series(close.values, index=close.index, dtype=float)

        common = close.index.intersection(nifty_close.index)
        if len(common) < 200:
            return None

        sw = close.loc[common].values.astype(float)
        nw = nifty_close.loc[common].values.astype(float)
        cp = float(sw[-1])

        if cp <= 0 or np.isnan(cp):
            return None

        def safe_ret(arr, back):
            if len(arr) > back and arr[-back] > 0:
                return float(arr[-1] / arr[-back] - 1)
            return 0.0

        ret_1m = safe_ret(sw, 22)
        ret_3m = safe_ret(sw, 63)
        ret_6m = safe_ret(sw, 126)
        ret_1y = safe_ret(sw, min(200, len(sw)-1))

        nifty_ret_1m = safe_ret(nw, 22)
        nifty_ret_3m = safe_ret(nw, 63)
        rs_1m = ret_1m - nifty_ret_1m
        rs_3m = ret_3m - nifty_ret_3m

        ma50  = float(np.mean(sw[-50:]))
        ma200 = float(np.mean(sw[-200:]))
        price_to_ma50  = cp / ma50  - 1 if ma50  > 0 else 0
        price_to_ma200 = cp / ma200 - 1 if ma200 > 0 else 0
        golden_cross   = 1 if ma50 > ma200 else 0

        daily_rets = np.diff(sw) / sw[:-1]
        daily_rets = daily_rets[~np.isnan(daily_rets)]
        vol_1m = float(np.std(daily_rets[-22:]) * np.sqrt(252)) if len(daily_rets) >= 22 else 0.3
        vol_3m = float(np.std(daily_rets[-63:]) * np.sqrt(252)) if len(daily_rets) >= 63 else 0.3

        high52 = float(np.max(sw[-252:])) if len(sw) >= 252 else float(np.max(sw))
        low52  = float(np.min(sw[-252:])) if len(sw) >= 252 else float(np.min(sw))
        rng    = high52 - low52
        pos52  = float((cp - low52) / rng) if rng > 0 else 0.5

        d     = np.diff(sw[-16:])
        gains = d[d > 0].mean() if len(d[d > 0]) > 0 else 0.001
        loss  = abs(d[d < 0].mean()) if len(d[d < 0]) > 0 else 0.001
        rsi   = float(100 - 100 / (1 + gains/loss))

        vol_trend = float(vol_1m / vol_3m) if vol_3m > 0 else 1.0

        return {
            'symbol':         sym,
            'current_price':  round(cp, 2),
            'ret_1m':         ret_1m,
            'ret_3m':         ret_3m,
            'ret_6m':         ret_6m,
            'ret_1y':         ret_1y,
            'rs_1m':          rs_1m,
            'rs_3m':          rs_3m,
            'price_to_ma50':  price_to_ma50,
            'price_to_ma200': price_to_ma200,
            'golden_cross':   golden_cross,
            'vol_1m':         vol_1m,
            'vol_3m':         vol_3m,
            'pos52':          pos52,
            'rsi':            rsi,
            'vol_trend':      vol_trend,
        }

    except Exception as e:
        print(f"    Error {sym}: {e}")
        return None


def run_predictions():
    print("\n" + "="*60)
    print("  NSE ML Screener — Live Predictions")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    print("\n Loading trained model...")
    saved    = joblib.load('ml_model.pkl')
    model    = saved['model']
    features = saved['features']
    accuracy = saved['accuracy']
    trained  = saved['trained']
    print(f"  Model accuracy: {accuracy*100:.1f}%")
    print(f"  Trained on:     {trained[:10]}")

    print("\n Downloading Nifty benchmark...")
    nifty_df = yf.download("^NSEI", period="2y", interval="1d",
                            auto_adjust=True, progress=False)
    if hasattr(nifty_df.columns, 'levels'):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_close = nifty_df['Close'].squeeze()
    nifty_close = pd.Series(nifty_close.values,
                             index=nifty_close.index, dtype=float)

    print(f"\n Computing features for {len(NSE_STOCKS)} stocks...")
    all_features = []
    for i, sym in enumerate(NSE_STOCKS, 1):
        print(f"  [{i:2d}/{len(NSE_STOCKS)}] {sym}...")
        feats = get_current_features(sym, nifty_close)
        if feats:
            all_features.append(feats)

    print(f"\n  Got features for {len(all_features)} stocks")

    print("\n Running ML predictions...")
    results = []
    for f in all_features:
        sym = f['symbol']
        X   = pd.DataFrame([{k: f[k] for k in features}])

        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])

        results.append({
            'symbol':       sym,
            'price':        f['current_price'],
            'ml_score':     round(prob * 100, 1),
            'prediction':   'OUTPERFORM' if pred == 1 else 'UNDERPERFORM',
            'ret_1m_pct':   round(f['ret_1m'] * 100, 1),
            'ret_3m_pct':   round(f['ret_3m'] * 100, 1),
            'rs_3m_pct':    round(f['rs_3m'] * 100, 1),
            'rsi':          round(f['rsi'], 1),
            'pos52':        round(f['pos52'] * 100, 1),
            'golden_cross': 'YES' if f['golden_cross'] else 'NO',
            'vol_3m_pct':   round(f['vol_3m'] * 100, 1),
        })

    results.sort(key=lambda x: x['ml_score'], reverse=True)

    print("\n" + "="*60)
    print("  TOP 10 — Most likely to outperform Nifty (3 months)")
    print("="*60)
    print(f"\n{'#':<4} {'SYMBOL':<14} {'PRICE':<12} {'ML SCORE':<12} "
          f"{'PREDICTION':<14} {'RSI':<8} {'52W POS'}")
    print("-"*70)

    for i, r in enumerate(results[:10], 1):
        print(f"{i:<4} {r['symbol']:<14} "
              f"Rs{r['price']:<11,.2f} "
              f"{r['ml_score']:<12.1f} "
              f"{r['prediction']:<14} "
              f"{r['rsi']:<8.1f} "
              f"{r['pos52']}%")

    print("\n" + "="*60)
    print("  BOTTOM 5 — Likely to underperform")
    print("="*60)
    for r in results[-5:]:
        print(f"  {r['symbol']:<14} Score: {r['ml_score']:.1f}  "
              f"RSI: {r['rsi']:.1f}  52W: {r['pos52']}%")

    df = pd.DataFrame(results)
    df.to_csv('ml_predictions.csv', index=False)
    print(f"\n Predictions saved to ml_predictions.csv")
    print(f" Model accuracy: {accuracy*100:.1f}% (vs 50% random chance)")

    return results


if __name__ == "__main__":
    run_predictions()
