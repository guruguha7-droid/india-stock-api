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
import time
import os
from datetime import datetime
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
    "GRAVITA", "DYCL", "TECHNOE",
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

    # Load Screener fundamentals
    screener_data = {}
    try:
        sdf = pd.read_csv('screener_fundamentals.csv')
        sdf = sdf[sdf['status'] == 'ok']
        screener_data = sdf.set_index('symbol').to_dict(orient='index')
        print(f"  Loaded Screener data for {len(screener_data)} stocks")
    except Exception:
        print("  Warning: screener_fundamentals.csv not found")

    # Download Nifty benchmark with retry
    print("\n Downloading Nifty benchmark...")
    nifty_df = None
    for attempt in range(3):
        try:
            nifty_df = yf.download("^NSEI", period="2y", interval="1d",
                                   auto_adjust=True, progress=False)
            if len(nifty_df) > 100:
                print(f"  Got {len(nifty_df)} days of Nifty data")
                break
            else:
                print(f"  Attempt {attempt+1} failed — retrying in 30s...")
                time.sleep(30)
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e} — retrying in 30s...")
            time.sleep(30)

    if nifty_df is None or len(nifty_df) < 100:
        print("Could not download Nifty data after 3 attempts.")
        print("   Wait 10 minutes and try again — Yahoo Finance rate limits heavy usage.")
        exit(1)

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

    # ── Fetch macro sentiment ─────────────────────────────────────────
    print("\n Fetching macro/economic sentiment...")
    from macro_sentiment import get_macro_sentiment, apply_macro_to_stock
    macro_data = get_macro_sentiment()
    print(f"  Fetched {len(macro_data)} macro topics")

    # ── Fetch news sentiment ──────────────────────────────────────────
    print("\n Fetching news sentiment for all 79 stocks (takes ~2 mins)...")
    from news_sentiment import get_sentiment_score
    sentiment_data = {}
    for i, f in enumerate(all_features, 1):
        sym = f['symbol']
        print(f"  [{i:2d}/{len(all_features)}] {sym}...", end=' ', flush=True)
        sent = get_sentiment_score(sym)
        sentiment_data[sym] = sent
        print(f"{sent['sentiment_label']} ({sent['sentiment_score']:+.0f})")
        time.sleep(0.8)

    # ── Fetch fundamentals ─────────────────────────────────────────────
    print("\n Fetching fundamentals for all stocks...")
    fundamentals = {}
    for f in all_features:
        sym = f['symbol']
        try:
            info = yf.Ticker(f"{sym}.NS").info
            fundamentals[sym] = {
                'pe':  float(info.get('trailingPE')    or 20),
                'pm':  float(info.get('profitMargins') or 0.10),
                'rg':  float(info.get('revenueGrowth') or 0.10),
                'eg':  float(info.get('earningsGrowth')or 0.10),
                'de':  float(info.get('debtToEquity')  or 50),
            }
        except Exception:
            fundamentals[sym] = {'pe':20,'pm':0.10,'rg':0.10,'eg':0.10,'de':50}

    # ── Run predictions ────────────────────────────────────────────────
    print("\n Running ML + Screener + yfinance predictions...")
    results = []
    for f in all_features:
        sym = f['symbol']
        X   = pd.DataFrame([{k: f[k] for k in features}])
        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        ml_raw = round(prob * 100, 1)

        # Screener data
        sc             = screener_data.get(sym, {})
        screener_score = float(sc.get('custom_score') or sc.get('investment_score', 50) or 50)
        screener_grade = sc.get('investment_grade', 'C') or 'C'
        roce           = float(sc.get('roce_latest_pct', 10) or 10)
        sales_cagr     = float(sc.get('sales_cagr_5y', 10) or 10)
        profit_cagr    = float(sc.get('profit_cagr_5y', 10) or 10)
        promoter       = float(sc.get('promoter_pct', 30) or 30)
        fcf_ok         = bool(sc.get('fcf_positive_3y', False))

        # yfinance fundamentals
        fd = fundamentals.get(sym, {})
        pe = fd.get('pe', 20)
        pm = fd.get('pm', 0.10)
        rg = fd.get('rg', 0.10)
        eg = fd.get('eg', 0.10)

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
        yfin_score = max(0, min(100, yfin_score))

        # ── News sentiment ────────────────────────────────────────────
        sent       = sentiment_data.get(sym, {})
        sent_raw   = float(sent.get('sentiment_score', 0) or 0)
        sent_label = sent.get('sentiment_label', 'neutral')
        sent_score = round(50 + sent_raw * 0.5, 1)
        sent_score = max(0, min(100, sent_score))

        # ── Macro sentiment for this stock ────────────────────────────
        macro       = apply_macro_to_stock(sym, macro_data)
        macro_raw   = float(macro.get('macro_score', 0) or 0)
        macro_label = macro.get('macro_label', 'neutral')
        macro_score = round(50 + macro_raw * 0.5, 1)
        macro_score = max(0, min(100, macro_score))

        # Gated sentiment — only include if signal is strong
        _sent_imp  = max(0, min(100, 50 + sent_raw * 0.5)) if abs(sent_raw) >= 15 else 0
        _macro_imp = max(0, min(100, 50 + macro_raw * 0.5)) if abs(macro_raw) >= 15 else 0
        if _sent_imp or _macro_imp:
            combined = round(
                ml_raw         * 0.22 +
                screener_score * 0.41 +
                yfin_score     * 0.25 +
                _sent_imp      * 0.07 +
                _macro_imp     * 0.05, 1)
        else:
            combined = round(
                ml_raw         * 0.25 +
                screener_score * 0.45 +
                yfin_score     * 0.30, 1)

        if combined >= 82:   inv_grade = 'A+'
        elif combined >= 68: inv_grade = 'A'
        elif combined >= 58: inv_grade = 'B'
        elif combined >= 48: inv_grade = 'C'
        else:                inv_grade = 'D'

        results.append({
            'symbol':         sym,
            'price':          f['current_price'],
            'ml_score':       ml_raw,
            'screener_score': screener_score,
            'screener_grade': screener_grade,
            'yfin_score':     round(yfin_score, 1),
            'combined':       combined,
            'inv_grade':      inv_grade,
            'roce':           roce,
            'sales_cagr_5y':  sales_cagr,
            'profit_cagr_5y': profit_cagr,
            'promoter_pct':   promoter,
            'fcf_positive':   fcf_ok,
            'pe':             round(pe, 1),
            'profit_margin':  round(pm * 100, 1),
            'rev_growth':     round(rg * 100, 1),
            'sentiment_score': sent_raw,
            'sentiment_label': sent_label,
            'macro_score':     macro_raw,
            'macro_label':     macro_label,
            'rsi':             round(f['rsi'], 1),
            'pos52':           round(f['pos52'] * 100, 1),
            'golden_cross':    bool(f['golden_cross']),
        })

    results.sort(key=lambda x: x['combined'], reverse=True)

    # Display
    print("\n" + "="*75)
    print("  TOP 10 — Combined ML + Screener + yfinance + Sentiment Score")
    print("="*75)
    print(f"\n{'#':<4}{'SYMBOL':<14}{'PRICE':<13}{'ML%':<8}{'SCR':<6}"
          f"{'SENT':<8}{'COMB':<8}{'GRD':<5}{'SENTIMENT'}")
    print("-"*75)
    for i, r in enumerate(results[:10], 1):
        sent_icon = 'POS' if r['sentiment_label'] == 'positive' else 'NEG' if r['sentiment_label'] == 'negative' else 'NEU'
        print(f"{i:<4}{r['symbol']:<14}"
              f"₹{r['price']:<12,.0f}"
              f"{r['ml_score']:<8.1f}"
              f"{r['screener_score']:<6.0f}"
              f"{r['sentiment_score']:+.0f}    "
              f"{r['combined']:<8.1f}"
              f"{r['inv_grade']:<5}"
              f"[{sent_icon}] {r['sentiment_label']}")

    print("\n" + "="*85)
    print("  BOTTOM 5 — Avoid these")
    print("="*85)
    for r in results[-5:]:
        print(f"  {r['symbol']:<14} ML:{r['ml_score']:.1f}% "
              f"Screener:{r['screener_score']:.0f} Grade:{r['screener_grade']} "
              f"Combined:{r['combined']:.1f}% [{r['inv_grade']}]")

    pd.DataFrame(results).to_csv('ml_predictions.csv', index=False)
    print(f"\n Saved to ml_predictions.csv")
    print(f" Model: {accuracy*100:.1f}% accuracy | Combined = 35% ML + 20% Screener + 15% yfinance + 15% Sentiment + 15% Macro")

    return results


if __name__ == "__main__":
    run_predictions()
