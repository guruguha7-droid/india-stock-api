"""
Nightly Cache Builder
=====================
Run this daily at market close (after 3:30 PM IST) to pre-compute:
  - ML features for all 139 stocks
  - Valuation (PE, EPS, dividend yield) from yfinance
  - Chart insights (revenue/profit/ROCE/debt trends) from Screener

Output: nightly_cache.json — read by api_server.py at request time

Usage:
    python nightly_cache.py

Schedule (cron example at 4 PM IST = 10:30 UTC):
    30 10 * * 1-5 cd /path/to/project && python nightly_cache.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT     = os.path.join(BASE_DIR, 'nightly_cache.json')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model.pkl')
CSV_PATH   = os.path.join(BASE_DIR, 'screener_fundamentals.csv')

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
    # Defence
    "COCHINSHIP", "MIDHANI", "DATAPATTNS", "PARAS",
    # Large Cap — Finance / Insurance / Aviation
    "INDUSINDBK", "SBILIFE", "HDFCLIFE", "LICI", "BAJAJHLDNG",
    "RECLTD", "PFC", "IRFC", "INDIGO",
    # Paints / Chemicals
    "PIDILITIND", "BERGEPAINT", "KANSAINER",
    # Mid Cap — IT / Electronics
    "HEXAWARE", "CYIENT", "MASTEK",
    "SONACOMS", "SYRMA", "KAYNES", "DIXON", "AMBER", "APLAPOLLO",
    # Mid Cap — Pharma / Healthcare
    "LAURUSLABS", "SOLARA", "SUVEN", "GLAND",
    "MEDANTA", "FORTIS", "RAINBOW", "KRSNAA", "METROPOLIS", "POLYMED",
    # Mid Cap — Consumer / FMCG
    "TRENT", "DMART", "ABFRL", "VSTIND", "GODFRYPHLP", "PGHH",
    "HONAUT", "WHIRLPOOL", "VOLTAS", "BLUESTARCO", "KAJARIACER", "CERA",
    # Mid Cap — Finance / Internet
    "ANGELONE", "CDSL", "BSE", "MCX",
    "NAUKRI", "POLICYBZR", "PAYTM", "NYKAA", "CARTRADE",
    # Mid Cap — Chemicals / Agri
    "NAVINFLUOR", "FLUOROCHEM", "DEEPAKNTR", "TATACHEM", "GNFC",
    "COROMANDEL", "PIIND", "RALLIS", "DHANUKA", "UPL", "CHAMBLFERT",
    # Mid Cap — Infrastructure
    "NCC", "PNCINFRA", "HGINFRA", "ASHOKA", "GPPL",
    # Mid Cap — Metals
    "GRAVITA", "DYCL",
    # Small Cap — Emerging
    "PGEL", "IDEAFORGE", "RATEGAIN", "EASEMYTRIP", "IXIGO", "YATHARTH",
    "HAPPYFORGE", "SANSERA", "CRAFTSMAN", "SUPRAJIT",
    "FINEORG", "GALAXYSURF", "CLEAN", "ROSSARI", "SUDARSCHEM",
    # Small Cap — Specialty Finance / Banking
    "IIFL", "CREDITACC", "UJJIVANSFB", "EQUITASBNK", "ESAFSFB", "UTKARSHBNK",
    "FUSION", "SPANDANA",
    # Hospitality
    "INDHOTEL", "LEMONTREE", "CHALET", "TAJGVK", "EIHOTEL",
    # Logistics
    "DELHIVERY", "BLUEDART", "GICRE", "NIACL", "CONCOR", "ALLCARGO", "MAHLOG",
    # Media
    "ZEEL", "SUNTV", "PVRINOX", "SAREGAMA",
    # Textiles
    "PAGEIND", "RAYMOND", "TRIDENT", "WELSPUNLIV", "KITEX",
]

SYMBOL_CSV_MAP = {'LTM': 'LTIM'}


def load_screener():
    if not os.path.exists(CSV_PATH):
        return {}
    df = pd.read_csv(CSV_PATH)
    df = df[df['status'] == 'ok']
    return df.set_index('symbol').to_dict(orient='index')


def compute_ml_features(symbol, sw, nw, fund):
    """Compute all 31 ML features from price arrays + fundamentals."""
    cp = float(sw[-1])
    if cp <= 0 or np.isnan(cp):
        return None

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

    eps_latest = float(fund.get('eps_latest') or 0) or None
    pe_ratio   = round(cp / eps_latest, 1) if eps_latest and eps_latest > 0 else 22.0
    eps_cagr   = float(fund.get('eps_cagr_5y') or 8.0)
    peg_ratio  = round(pe_ratio / max(eps_cagr, 0.1), 2)

    return {
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
        'roce_latest_pct':  float(fund.get('roce_latest_pct')  or 12.0),
        'opm_latest_pct':   float(fund.get('opm_latest_pct')   or 12.0),
        'sales_cagr_5y':    float(fund.get('sales_cagr_5y')    or 10.0),
        'profit_cagr_5y':   float(fund.get('profit_cagr_5y')   or 8.0),
        'eps_cagr_5y':      eps_cagr,
        'sales_growth_1y':  float(fund.get('sales_growth_1y')  or 8.0),
        'profit_growth_1y': float(fund.get('profit_growth_1y') or 8.0),
        'opm_trend_5y':     float(fund.get('opm_trend_5y')     or 0.0),
        'roce_trend_5y':    float(fund.get('roce_trend_5y')    or 0.0),
        'promoter_pct':     float(fund.get('promoter_pct')     or 45.0),
        'fii_pct':          float(fund.get('fii_pct')          or 15.0),
        'fcf_positive_3y':  float(bool(fund.get('fcf_positive_3y'))),
        'debt_reducing':    float(bool(fund.get('debt_reducing'))),
        'screener_de':      float(fund.get('screener_de')       or 50.0),
        'pe_ratio':         pe_ratio,
        'pb_ratio':         3.0,
        'peg_ratio':        peg_ratio,
        # Extra fields for response
        '_price':           round(cp, 2),
        '_ret_1m_pct':      round(ret_1m * 100, 1),
        '_ret_3m_pct':      round(ret_3m * 100, 1),
        '_pos52_pct':       round(float((cp - l52) / rng) * 100 if rng > 0 else 50, 1),
        '_golden_cross':    bool(ma50 > ma200),
    }


def compute_valuation(symbol, price, fund):
    """Compute valuation metrics from Screener CSV."""
    # NOTE: eps_latest is annual EPS from Screener, not TTM.
    # For companies with strong recent quarters, fair value may be understated.
    eps_latest    = float(fund.get('eps_latest') or 0) or None
    pe            = round(price / eps_latest, 1) if eps_latest and eps_latest > 0 and price else None
    div_payout    = float(fund.get('dividend_payout_pct') or 0)
    div_per_share = round(div_payout / 100 * eps_latest, 2) if eps_latest and div_payout > 0 else None
    div_yield     = round(div_per_share / price, 6) if div_per_share and price and price > 0 else None

    # PB: derive shares from profit_cr / eps_latest, then compute mcap/book
    pb = None
    try:
        nw_cr     = float(fund.get('networth_cr') or 0)
        profit_cr = float(fund.get('profit_latest_cr') or 0)
        if nw_cr > 0 and profit_cr > 0 and eps_latest and eps_latest > 0:
            shares   = (profit_cr * 1e7) / eps_latest
            pb       = round((price * shares) / (nw_cr * 1e7), 2)
    except Exception:
        pass

    # Profit margin from OPM
    pm = None
    try:
        opm = float(fund.get('opm_latest_pct') or 0)
        if opm:
            pm = round(opm / 100, 4)
    except Exception:
        pass

    # Revenue growth from 1Y sales growth
    rg = None
    try:
        sg = float(fund.get('sales_growth_1y') or 0)
        if sg:
            rg = round(sg / 100, 4)
    except Exception:
        pass

    return {
        'pe_ratio':        pe,
        'eps':             eps_latest,
        'dividend_yield':  div_yield,
        'pb_ratio':        pb,
        'profit_margin':   pm,
        'revenue_growth':  rg,
        'earnings_growth': round(float(fund.get('eps_growth_1y') or 0) / 100, 4) if fund.get('eps_growth_1y') else None,
        'debt_to_equity':  float(fund.get('screener_de') or 0) or None,
    }


def compute_chart_insights(fund):
    """Generate chart trend descriptions from Screener CSV data."""
    def trend_desc(vals, higher_is_good=True):
        if not vals or len(vals) < 3:
            return None
        recent  = vals[-3:]
        older   = vals[:3]
        avg_rec = sum(recent) / len(recent)
        avg_old = sum(older)  / len(older)
        if avg_old == 0:
            return None
        chg = (avg_rec - avg_old) / abs(avg_old) * 100

        if higher_is_good:
            if chg > 20:    trend, quality = "strong uptrend", "good"
            elif chg > 5:   trend, quality = "gradual uptrend", "good"
            elif chg > -5:  trend, quality = "relatively flat", "neutral"
            elif chg > -20: trend, quality = "gradual decline", "bad"
            else:           trend, quality = "sharp decline", "bad"
        else:
            if chg > 20:    trend, quality = "sharp increase", "bad"
            elif chg > 5:   trend, quality = "gradual increase", "bad"
            elif chg > -5:  trend, quality = "relatively flat", "neutral"
            elif chg > -20: trend, quality = "gradual reduction", "good"
            else:           trend, quality = "sharp reduction", "good"

        return {"trend": trend, "quality": quality, "change_pct": round(chg, 1)}

    # Build approximate multi-year arrays from available CSV fields
    sales_cagr  = float(fund.get('sales_cagr_5y')  or 0)
    profit_cagr = float(fund.get('profit_cagr_5y') or 0)
    roce_latest = float(fund.get('roce_latest_pct') or 0)
    roce_avg    = float(fund.get('roce_avg_5y')    or roce_latest)
    roce_trend  = float(fund.get('roce_trend_5y')  or 0)
    debt_growth = float(fund.get('debt_growth_1y') or 0)

    insights = {}

    # Revenue trend from CAGR
    if sales_cagr != 0:
        q = "good" if sales_cagr > 10 else "bad" if sales_cagr < 0 else "neutral"
        t = "strong uptrend" if sales_cagr > 15 else "gradual uptrend" if sales_cagr > 5 else "relatively flat" if sales_cagr > -5 else "gradual decline"
        insights['revenue'] = {
            "trend": t, "quality": q, "change_pct": round(sales_cagr, 1),
            "summary": f"Revenue 5Y CAGR of {sales_cagr:.1f}% — {'strong business growth' if q=='good' else 'revenue declining, watch closely' if q=='bad' else 'stable but limited growth'}."
        }

    # Profit trend from CAGR
    if profit_cagr != 0:
        q = "good" if profit_cagr > 10 else "bad" if profit_cagr < 0 else "neutral"
        t = "strong uptrend" if profit_cagr > 15 else "gradual uptrend" if profit_cagr > 5 else "relatively flat" if profit_cagr > -5 else "gradual decline"
        insights['profit'] = {
            "trend": t, "quality": q, "change_pct": round(profit_cagr, 1),
            "summary": f"Profit 5Y CAGR of {profit_cagr:.1f}% — {'earnings expanding well' if q=='good' else 'profitability under pressure' if q=='bad' else 'margins holding steady'}."
        }

    # ROCE trend
    if roce_latest != 0:
        q = "good" if roce_trend > 0 else "bad" if roce_trend < -3 else "neutral"
        t = "gradual uptrend" if roce_trend > 2 else "gradual decline" if roce_trend < -2 else "relatively flat"
        insights['roce'] = {
            "trend": t, "quality": q, "change_pct": round(roce_trend, 1),
            "summary": f"ROCE at {roce_latest:.1f}% (5Y avg {roce_avg:.1f}%) — {'improving capital efficiency' if q=='good' else 'declining returns on capital, monitor closely' if q=='bad' else 'capital efficiency stable'}."
        }

    # Debt trend
    if debt_growth != 0:
        q = "good" if debt_growth < 0 else "bad" if debt_growth > 15 else "neutral"
        t = "sharp reduction" if debt_growth < -10 else "gradual reduction" if debt_growth < 0 else "gradual increase" if debt_growth < 15 else "sharp increase"
        insights['debt'] = {
            "trend": t, "quality": q, "change_pct": round(debt_growth, 1),
            "summary": f"Debt {'reduced' if debt_growth < 0 else 'grew'} {abs(debt_growth):.1f}% YoY — {'balance sheet strengthening' if q=='good' else 'rising debt increases financial risk' if q=='bad' else 'debt levels stable'}."
        }

    return insights


def build_cache():
    print("=" * 60)
    print(f"  Nightly Cache Builder — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load model
    print("\nLoading ML model...")
    saved    = joblib.load(MODEL_PATH)
    model    = saved['model']
    features = saved['features']
    accuracy = saved['accuracy']
    print(f"  Model: {saved.get('model_name','?')} — {accuracy*100:.1f}% accuracy")

    # Load Screener fundamentals
    print("\nLoading Screener fundamentals...")
    screener = load_screener()
    print(f"  Loaded {len(screener)} stocks")

    # Download Nifty
    print("\nDownloading Nifty benchmark...")
    nifty_df = yf.download("^NSEI", period="2y", interval="1d",
                           auto_adjust=True, progress=False)
    if hasattr(nifty_df.columns, 'levels'):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_close = nifty_df['Close'].squeeze()
    nifty_arr   = np.array(nifty_close.values, dtype=float)
    print(f"  Got {len(nifty_arr)} days")

    cache = {
        'built_at': datetime.now().isoformat(),
        'accuracy': round(accuracy * 100, 1),
        'stocks':   {}
    }

    print(f"\nProcessing {len(NSE_STOCKS)} stocks...")
    ok, failed = 0, []

    for i, sym in enumerate(NSE_STOCKS, 1):
        try:
            csv_sym = SYMBOL_CSV_MAP.get(sym, sym)
            fund    = screener.get(csv_sym, screener.get(sym, {}))

            # Download price data
            df = yf.download(f"{sym}.NS", period="2y", interval="1d",
                             auto_adjust=True, progress=False)
            if df is None or len(df) < 150:
                failed.append(sym)
                print(f"  [{i:3d}] FAIL {sym} — insufficient data")
                time.sleep(0.3)
                continue

            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)

            close_s = pd.Series(df['Close'].squeeze().values,
                                index=df.index, dtype=float)

            # Align with Nifty
            nifty_s  = pd.Series(nifty_arr, index=nifty_close.index, dtype=float)
            common   = close_s.index.intersection(nifty_s.index)
            if len(common) >= 100:
                sw = close_s.loc[common].values.astype(float)
                nw = nifty_s.loc[common].values.astype(float)
            else:
                sw = close_s.values.astype(float)
                nw = None

            # Compute ML features
            f = compute_ml_features(sym, sw, nw, fund)
            if f is None:
                failed.append(sym)
                print(f"  [{i:3d}] FAIL {sym} — feature computation failed")
                time.sleep(0.3)
                continue

            # Run ML prediction
            X    = pd.DataFrame([{k: f[k] for k in features}])
            prob = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])

            price = f['_price']

            # Compute valuation from CSV
            val = compute_valuation(sym, price, fund)

            # Compute chart insights from CSV
            insights = compute_chart_insights(fund)

            cache['stocks'][sym] = {
                'ml': {
                    'ml_score':     round(prob * 100, 1),
                    'prediction':   'OUTPERFORM' if pred == 1 else 'UNDERPERFORM',
                    'accuracy':     round(accuracy * 100, 1),
                    'rsi':          round(f['rsi'], 1),
                    'pos52_pct':    f['_pos52_pct'],
                    'ret_1m_pct':   f['_ret_1m_pct'],
                    'ret_3m_pct':   f['_ret_3m_pct'],
                    'ret_1y_pct':   round(f['ret_1y'] * 100, 1),
                    'golden_cross': f['_golden_cross'],
                },
                'valuation':      val,
                'chart_insights': insights,
                'cached_price':   price,
            }

            ok += 1
            print(f"  [{i:3d}] OK  {sym:<15} ML:{prob*100:.1f}%  RSI:{f['rsi']:.0f}  PE:{val['pe_ratio']}")
            time.sleep(0.4)  # be gentle with Yahoo Finance

        except Exception as e:
            failed.append(sym)
            print(f"  [{i:3d}] ERR {sym} — {e}")
            time.sleep(0.5)

    print(f"\nDone — {ok} stocks cached, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")

    # Save
    with open(OUTPUT, 'w') as fp:
        json.dump(cache, fp, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}")
    print(f"Cache size: {os.path.getsize(OUTPUT)/1024:.1f} KB")

    # ── Auto-commit cache to GitHub so it survives Render restarts ────
    try:
        import subprocess
        subprocess.run(['git', 'config', 'user.email', 'cache@graham.app'], check=False, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Graham Cache Bot'], check=False, capture_output=True)
        subprocess.run(['git', 'add', 'nightly_cache.json'], check=False, capture_output=True)
        subprocess.run(['git', 'commit', '-m', f'cache: nightly rebuild {datetime.now().strftime("%Y-%m-%d %H:%M")}'], check=False, capture_output=True)
        subprocess.run(['git', 'push'], check=False, capture_output=True)
        print("  Cache committed to GitHub")
    except Exception as e:
        print(f"  Git push failed: {e}")

    return cache


if __name__ == '__main__':
    build_cache()
