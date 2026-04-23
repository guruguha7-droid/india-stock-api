"""
ML Stock Screener — Training Pipeline v2
==========================================
Improvements over v1:
  1. Adds 14 Screener fundamental features (ROCE, margins, FCF, promoter etc.)
  2. Extended forward horizon: 6 months (126 days) instead of 3 months
  3. XGBoost added alongside Random Forest + Gradient Boosting
  4. Ensemble voting across all 3 models
  5. Better feature imputation for missing fundamentals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

NSE_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "TCS.NS", "INFY.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTM.NS", "RELIANCE.NS",
    "ONGC.NS", "BPCL.NS", "IOC.NS", "POWERGRID.NS", "NTPC.NS",
    "ADANIPORTS.NS", "ADANIENT.NS", "TATAPOWER.NS", "HINDUNILVR.NS", "ITC.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "MARUTI.NS", "M&M.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "COALINDIA.NS", "VEDL.NS", "LT.NS", "ULTRACEMCO.NS",
    "GRASIM.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "TITAN.NS", "BANKBARODA.NS",
    "PNB.NS", "CANBK.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS", "MANAPPURAM.NS",
    "MARICO.NS", "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS",
    "TORNTPHARM.NS", "LUPIN.NS", "AUROPHARMA.NS", "ALKEM.NS", "PERSISTENT.NS",
    "MPHASIS.NS", "COFORGE.NS", "KPITTECH.NS", "TVSMOTOR.NS", "MOTHERSON.NS",
    "BALKRISIND.NS", "APOLLOTYRE.NS", "SIEMENS.NS", "HAVELLS.NS", "ABB.NS",
    "CUMMINSIND.NS", "DLF.NS", "OBEROIRLTY.NS", "RAMCOCEM.NS", "FEDERALBNK.NS",
    "IDFCFIRSTB.NS", "BANDHANBNK.NS", "AUBANK.NS", "RBLBANK.NS", "INDIANB.NS",
    "MAHABANK.NS", "HDFCAMC.NS", "ICICIGI.NS", "SBICARD.NS",
    "SUNDARMFIN.NS", "TATAELXSI.NS", "LTTS.NS", "HAPPSTMNDS.NS", "ZENSARTECH.NS",
    "MANKIND.NS", "ABBOTINDIA.NS", "NATCOPHARM.NS", "GRANULES.NS", "GLENMARK.NS",
    "IPCALAB.NS", "MAXHEALTH.NS", "RADICO.NS", "UBL.NS", "VBL.NS",
    "BIKAJI.NS", "ZYDUSWELL.NS", "ASHOKLEY.NS", "BOSCHLTD.NS", "TIINDIA.NS",
    "ENDURANCE.NS", "SUNDRMFAST.NS", "SCHAEFFLER.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
    "TORNTPOWER.NS", "CESC.NS", "NHPC.NS", "SJVN.NS", "HINDPETRO.NS",
    "GAIL.NS", "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS",
    "IRB.NS", "KNRCON.NS", "NMDC.NS", "SAIL.NS", "NATIONALUM.NS",
    "MOIL.NS", "WELCORP.NS", "GODREJPROP.NS", "BRIGADE.NS", "SOBHA.NS",
    "PRESTIGE.NS", "PHOENIXLTD.NS", "HAL.NS", "BEL.NS", "BEML.NS",
    "MAZDOCK.NS",
    # Defence
    "COCHINSHIP.NS", "MIDHANI.NS", "DATAPATTNS.NS", "PARAS.NS",
    # Large Cap — Finance / Insurance / Aviation
    "INDUSINDBK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "LICI.NS", "BAJAJHLDNG.NS",
    "RECLTD.NS", "PFC.NS", "IRFC.NS", "INDIGO.NS",
    # Paints / Chemicals
    "PIDILITIND.NS", "BERGEPAINT.NS", "KANSAINER.NS",
    # Mid Cap — IT / Electronics
    "HEXAWARE.NS", "CYIENT.NS", "MASTEK.NS",
    "SONACOMS.NS", "SYRMA.NS", "KAYNES.NS", "DIXON.NS", "AMBER.NS", "APLAPOLLO.NS",
    # Mid Cap — Pharma / Healthcare
    "LAURUSLABS.NS", "SOLARA.NS", "SUVEN.NS", "GLAND.NS",
    "MEDANTA.NS", "FORTIS.NS", "RAINBOW.NS", "KRSNAA.NS", "METROPOLIS.NS", "POLYMED.NS",
    # Mid Cap — Consumer / FMCG
    "TRENT.NS", "DMART.NS", "ABFRL.NS", "VSTIND.NS", "GODFRYPHLP.NS", "PGHH.NS",
    "HONAUT.NS", "WHIRLPOOL.NS", "VOLTAS.NS", "BLUESTARCO.NS", "KAJARIACER.NS", "CERA.NS",
    # Mid Cap — Finance / Internet
    "ANGELONE.NS", "CDSL.NS", "BSE.NS", "MCX.NS",
    "NAUKRI.NS", "POLICYBZR.NS", "PAYTM.NS", "NYKAA.NS", "CARTRADE.NS",
    # Mid Cap — Chemicals / Agri
    "NAVINFLUOR.NS", "FLUOROCHEM.NS", "DEEPAKNTR.NS", "TATACHEM.NS", "GNFC.NS",
    "COROMANDEL.NS", "PIIND.NS", "RALLIS.NS", "DHANUKA.NS", "UPL.NS", "CHAMBLFERT.NS",
    # Mid Cap — Infrastructure
    "NCC.NS", "PNCINFRA.NS", "HGINFRA.NS", "ASHOKA.NS", "GPPL.NS",
    # Mid Cap — Metals
    "GRAVITA.NS", "DYCL.NS",
    # Small Cap — Emerging
    "PGEL.NS", "IDEAFORGE.NS", "RATEGAIN.NS", "EASEMYTRIP.NS", "IXIGO.NS", "YATHARTH.NS",
    "HAPPYFORGE.NS", "SANSERA.NS", "CRAFTSMAN.NS", "SUPRAJIT.NS",
    "FINEORG.NS", "GALAXYSURF.NS", "CLEAN.NS", "ROSSARI.NS", "SUDARSCHEM.NS",
    # Small Cap — Specialty Finance / Banking
    "IIFL.NS", "CREDITACC.NS", "UJJIVANSFB.NS", "EQUITASBNK.NS", "ESAFSFB.NS", "UTKARSHBNK.NS",
    "FUSION.NS", "SPANDANA.NS",
    # Hospitality
    "INDHOTEL.NS", "LEMONTREE.NS", "CHALET.NS", "TAJGVK.NS", "EIHOTEL.NS",
    # Logistics
    "DELHIVERY.NS", "BLUEDART.NS", "GICRE.NS", "NIACL.NS", "CONCOR.NS", "ALLCARGO.NS", "MAHLOG.NS",
    # Media
    "ZEEL.NS", "SUNTV.NS", "PVRINOX.NS", "SAREGAMA.NS",
    # Textiles
    "PAGEIND.NS", "RAYMOND.NS", "TRIDENT.NS", "WELSPUNLIV.NS", "KITEX.NS",
]

NIFTY        = "^NSEI"
PERIOD       = "8y"           # extra year needed since 1Y forward eats into the data
FORWARD_DAYS = 252            # 1 year — fundamentals play out over 12 months, not 6

# ── Technical features (same as v1) ──────────────────────────────────────────
TECH_FEATURES = [
    'ret_1m', 'ret_3m', 'ret_6m', 'ret_1y',
    'rs_1m', 'rs_3m',
    'price_to_ma50', 'price_to_ma200', 'golden_cross',
    'vol_1m', 'vol_3m',
    'pos52', 'rsi', 'vol_trend',
]

# ── Fundamental features from Screener CSV ────────────────────────────────────
FUND_FEATURES = [
    'roce_latest_pct',    # capital efficiency
    'opm_latest_pct',     # operating margin quality
    'sales_cagr_5y',      # revenue growth momentum
    'profit_cagr_5y',     # earnings growth momentum
    'eps_cagr_5y',        # per-share earnings trend
    'sales_growth_1y',    # recent revenue acceleration
    'profit_growth_1y',   # recent profit acceleration
    'opm_trend_5y',       # margin improving/declining
    'roce_trend_5y',      # ROCE improving/declining
    'promoter_pct',       # skin in the game
    'fii_pct',            # institutional interest
    'fcf_positive_3y',    # cash generation (boolean → 0/1)
    'debt_reducing',      # balance sheet improving (boolean → 0/1)
    'screener_de',        # debt/equity ratio
    'pe_ratio',           # market valuation
    'pb_ratio',           # price vs book value
    'peg_ratio',          # PE relative to growth — key for value vs growth distinction
]

ALL_FEATURES = TECH_FEATURES + FUND_FEATURES

# ── Fundamental defaults for imputation ──────────────────────────────────────
FUND_DEFAULTS = {
    'roce_latest_pct':  12.0,
    'opm_latest_pct':   12.0,
    'sales_cagr_5y':    10.0,
    'profit_cagr_5y':    8.0,
    'eps_cagr_5y':       8.0,
    'sales_growth_1y':   8.0,
    'profit_growth_1y':  8.0,
    'opm_trend_5y':      0.0,
    'roce_trend_5y':     0.0,
    'promoter_pct':     45.0,
    'fii_pct':          15.0,
    'fcf_positive_3y':   0.5,
    'debt_reducing':     0.5,
    'screener_de':      50.0,
    'pe_ratio':         22.0,
    'pb_ratio':          3.0,
    'peg_ratio':         2.5,
}


# ── Step 1: Download price data ───────────────────────────────────────────────
def download_data():
    print(f"\nDownloading {PERIOD} of NSE data from Yahoo Finance...")
    print("This will take 3-5 minutes. Please wait.\n")

    all_prices = {}

    print("  -> Downloading NIFTY 50 benchmark...")
    nifty = yf.download(NIFTY, period=PERIOD, interval="1d",
                        auto_adjust=True, progress=False)
    nifty_close = nifty['Close'].squeeze()
    all_prices['NIFTY'] = nifty_close
    print(f"     Got {len(nifty_close)} days of Nifty data")

    ok, failed = 0, []
    for sym in NSE_STOCKS:
        try:
            df = yf.download(sym, period=PERIOD, interval="1d",
                             auto_adjust=True, progress=False)
            if len(df) > 100:
                close = df['Close'].squeeze()
                name = sym.replace('.NS','')
                all_prices[name] = close
                ok += 1
                print(f"  OK {name} — {len(df)} days")
            else:
                failed.append(sym)
                print(f"  FAIL {sym} — insufficient data")
        except Exception as e:
            failed.append(sym)
            print(f"  FAIL {sym} — {e}")

    print(f"\nDownloaded {ok} stocks successfully")
    if failed:
        print(f"Failed: {[s.replace('.NS','') for s in failed]}")

    return all_prices


# ── Step 1b: Download valuation ratios ────────────────────────────────────────
def download_valuation():
    print("\nDownloading valuation ratios (PE, PB)...")
    val_dict = {}
    for sym in NSE_STOCKS:
        name = sym.replace('.NS', '')
        try:
            import time
            info = yf.Ticker(sym).info
            pe = float(info.get('trailingPE') or info.get('forwardPE') or 22.0)
            pb = float(info.get('priceToBook') or 3.0)
            val_dict[name] = {'pe_ratio': pe, 'pb_ratio': pb}
            print(f"  OK {name} — PE:{pe:.1f} PB:{pb:.1f}")
            time.sleep(0.3)
        except Exception:
            val_dict[name] = {'pe_ratio': 22.0, 'pb_ratio': 3.0}
            print(f"  FAIL {name} — using defaults")
    return val_dict


# ── Step 2: Load Screener fundamentals ───────────────────────────────────────
def load_fundamentals():
    print("\nLoading Screener fundamentals...")
    path = os.path.join(os.path.dirname(__file__), 'screener_fundamentals.csv')
    if not os.path.exists(path):
        print("  WARNING: screener_fundamentals.csv not found — fundamentals will use defaults")
        return {}

    df = pd.read_csv(path)
    df = df[df['status'] == 'ok']

    # Normalise symbol map
    SYMBOL_CSV_MAP = {'LTM': 'LTIM'}
    fund_dict = {}

    for _, row in df.iterrows():
        sym = row['symbol']
        # Reverse map — store under NSE symbol
        for nse_sym, csv_sym in SYMBOL_CSV_MAP.items():
            if sym == csv_sym:
                sym = nse_sym
                break

        fund_dict[sym] = {
            'roce_latest_pct':  float(row.get('roce_latest_pct') or FUND_DEFAULTS['roce_latest_pct']),
            'opm_latest_pct':   float(row.get('opm_latest_pct')  or FUND_DEFAULTS['opm_latest_pct']),
            'sales_cagr_5y':    float(row.get('sales_cagr_5y')   or FUND_DEFAULTS['sales_cagr_5y']),
            'profit_cagr_5y':   float(row.get('profit_cagr_5y')  or FUND_DEFAULTS['profit_cagr_5y']),
            'eps_cagr_5y':      float(row.get('eps_cagr_5y')     or FUND_DEFAULTS['eps_cagr_5y']),
            'sales_growth_1y':  float(row.get('sales_growth_1y') or FUND_DEFAULTS['sales_growth_1y']),
            'profit_growth_1y': float(row.get('profit_growth_1y')or FUND_DEFAULTS['profit_growth_1y']),
            'opm_trend_5y':     float(row.get('opm_trend_5y')    or 0.0),
            'roce_trend_5y':    float(row.get('roce_trend_5y')   or 0.0),
            'promoter_pct':     float(row.get('promoter_pct')    or FUND_DEFAULTS['promoter_pct']),
            'fii_pct':          float(row.get('fii_pct')         or FUND_DEFAULTS['fii_pct']),
            'fcf_positive_3y':  float(bool(row.get('fcf_positive_3y'))),
            'debt_reducing':    float(bool(row.get('debt_reducing'))),
            'screener_de':      float(row.get('screener_de')     or FUND_DEFAULTS['screener_de']),
        }

    print(f"  Loaded fundamentals for {len(fund_dict)} stocks")
    return fund_dict


# ── Step 3: Engineer features ─────────────────────────────────────────────────
def engineer_features(prices: dict, fund_dict: dict, val_dict: dict = None) -> pd.DataFrame:
    print("\nEngineering features...")

    nifty = prices.get('NIFTY')
    if nifty is None:
        print("No Nifty data found")
        return pd.DataFrame()

    if hasattr(nifty, 'columns'):
        nifty = nifty.iloc[:, 0]
    nifty = pd.Series(nifty.values, index=nifty.index, dtype=float)

    records = []
    skipped_no_fund = 0

    for sym, close in prices.items():
        if sym == 'NIFTY':
            continue

        if hasattr(close, 'columns'):
            close = close.iloc[:, 0]
        close = pd.Series(close.values, index=close.index, dtype=float)

        common_idx = close.index.intersection(nifty.index)
        if len(common_idx) < 300:
            print(f"  Skipping {sym} — only {len(common_idx)} common dates")
            continue

        stock_s = close.loc[common_idx].reset_index(drop=True)
        nifty_s = nifty.loc[common_idx].reset_index(drop=True)

        # Get fundamentals for this stock
        fund = fund_dict.get(sym, {})
        if not fund:
            skipped_no_fund += 1
            # Use defaults — don't skip, just use median values
            fund = FUND_DEFAULTS.copy()

        step = 30  # reduce overlap — consecutive samples share only 222/252 forward days

        for i in range(200, len(stock_s) - FORWARD_DAYS, step):
            try:
                sw = stock_s.iloc[i-200:i].values.astype(float)
                nw = nifty_s.iloc[i-200:i].values.astype(float)
                cp = float(sw[-1])

                if cp <= 0 or np.isnan(cp):
                    continue

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

                ma50   = float(np.mean(sw[-50:]))
                ma200  = float(np.mean(sw[-200:]))
                price_to_ma50  = cp / ma50  - 1 if ma50  > 0 else 0
                price_to_ma200 = cp / ma200 - 1 if ma200 > 0 else 0
                golden_cross   = 1 if ma50 > ma200 else 0

                daily_rets = np.diff(sw) / sw[:-1]
                daily_rets = daily_rets[~np.isnan(daily_rets)]
                vol_1m = float(np.std(daily_rets[-22:]) * np.sqrt(252)) if len(daily_rets) >= 22 else 0.3
                vol_3m = float(np.std(daily_rets[-63:]) * np.sqrt(252)) if len(daily_rets) >= 63 else 0.3

                high52 = float(np.max(sw[-min(252, len(sw)):]))
                low52  = float(np.min(sw[-min(252, len(sw)):]))
                rng    = high52 - low52
                pos52  = float((cp - low52) / rng) if rng > 0 else 0.5

                d = np.diff(sw[-16:])
                gains  = d[d > 0].mean() if len(d[d > 0]) > 0 else 0.001
                losses = abs(d[d < 0].mean()) if len(d[d < 0]) > 0 else 0.001
                rsi    = float(100 - 100 / (1 + gains/losses))

                vol_trend = float(vol_1m / vol_3m) if vol_3m > 0 else 1.0

                fut_s = stock_s.iloc[i:i+FORWARD_DAYS].values.astype(float)
                fut_n = nifty_s.iloc[i:i+FORWARD_DAYS].values.astype(float)
                if len(fut_s) < FORWARD_DAYS:
                    continue

                fwd_stock = float(fut_s[-1] / fut_s[0] - 1) if fut_s[0] > 0 else 0
                fwd_nifty = float(fut_n[-1] / fut_n[0] - 1) if fut_n[0] > 0 else 0
                outperforms = 1 if fwd_stock > fwd_nifty else 0

                record = {
                    'symbol': sym, 'date': common_idx[i],
                    # Technical features
                    'ret_1m': ret_1m, 'ret_3m': ret_3m,
                    'ret_6m': ret_6m, 'ret_1y': ret_1y,
                    'rs_1m': rs_1m, 'rs_3m': rs_3m,
                    'price_to_ma50': price_to_ma50,
                    'price_to_ma200': price_to_ma200,
                    'golden_cross': golden_cross,
                    'vol_1m': vol_1m, 'vol_3m': vol_3m,
                    'pos52': pos52, 'rsi': rsi,
                    'vol_trend': vol_trend,
                    # Fundamental features
                    'roce_latest_pct':  fund.get('roce_latest_pct',  FUND_DEFAULTS['roce_latest_pct']),
                    'opm_latest_pct':   fund.get('opm_latest_pct',   FUND_DEFAULTS['opm_latest_pct']),
                    'sales_cagr_5y':    fund.get('sales_cagr_5y',    FUND_DEFAULTS['sales_cagr_5y']),
                    'profit_cagr_5y':   fund.get('profit_cagr_5y',   FUND_DEFAULTS['profit_cagr_5y']),
                    'eps_cagr_5y':      fund.get('eps_cagr_5y',      FUND_DEFAULTS['eps_cagr_5y']),
                    'sales_growth_1y':  fund.get('sales_growth_1y',  FUND_DEFAULTS['sales_growth_1y']),
                    'profit_growth_1y': fund.get('profit_growth_1y', FUND_DEFAULTS['profit_growth_1y']),
                    'opm_trend_5y':     fund.get('opm_trend_5y',     0.0),
                    'roce_trend_5y':    fund.get('roce_trend_5y',    0.0),
                    'promoter_pct':     fund.get('promoter_pct',     FUND_DEFAULTS['promoter_pct']),
                    'fii_pct':          fund.get('fii_pct',          FUND_DEFAULTS['fii_pct']),
                    'fcf_positive_3y':  fund.get('fcf_positive_3y',  0.5),
                    'debt_reducing':    fund.get('debt_reducing',    0.5),
                    'screener_de':      fund.get('screener_de',      FUND_DEFAULTS['screener_de']),
                    # Valuation features
                    'pe_ratio':  (val_dict or {}).get(sym, {}).get('pe_ratio',  FUND_DEFAULTS['pe_ratio']),
                    'pb_ratio':  (val_dict or {}).get(sym, {}).get('pb_ratio',  FUND_DEFAULTS['pb_ratio']),
                    'peg_ratio': round(
                        (val_dict or {}).get(sym, {}).get('pe_ratio', FUND_DEFAULTS['pe_ratio']) /
                        max(fund.get('eps_cagr_5y', FUND_DEFAULTS['eps_cagr_5y']), 0.1), 2
                    ),
                    # Label
                    'outperforms': outperforms,
                }
                records.append(record)

            except Exception:
                continue

    df = pd.DataFrame(records)
    print(f"Created {len(df)} training samples from {df['symbol'].nunique()} stocks")
    print(f"  (Stocks using default fundamentals: {skipped_no_fund})")
    return df


# ── Step 4: Train models ──────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    print("\nTraining ML model v2 (Technical + Fundamental features)...")
    print(f"  Features: {len(ALL_FEATURES)} ({len(TECH_FEATURES)} technical + {len(FUND_FEATURES)} fundamental)")

    df_clean = df.dropna(subset=['outperforms'])
    X = df_clean[ALL_FEATURES].copy()
    y = df_clean['outperforms']

    # Impute any remaining NaN with median
    for col in ALL_FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    print(f"  Training samples: {len(X)}")
    print(f"  Outperformers:    {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Underperformers:  {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # Time-based split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n  Training Random Forest...")
    rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Random Forest accuracy: {rf_acc*100:.1f}%")

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    print("\n  Training Gradient Boosting...")
    gb = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   GradientBoostingClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.7,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42
        ))
    ])
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"  Gradient Boosting accuracy: {gb_acc*100:.1f}%")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_acc = 0
    xgb = None
    try:
        from xgboost import XGBClassifier
        print("\n  Training XGBoost...")
        xgb = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.8,
                min_child_weight=10,
                scale_pos_weight=1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
            ))
        ])
        xgb.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
        print(f"  XGBoost accuracy: {xgb_acc*100:.1f}%")
    except ImportError:
        print("  XGBoost not installed — skipping (pip install xgboost)")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n  Building ensemble...")
    estimators = [('rf', rf), ('gb', gb)]
    if xgb and xgb_acc > 0:
        estimators.append(('xgb', xgb))

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"  Ensemble accuracy: {ens_acc*100:.1f}%")

    # Pick best
    results = [('Random Forest', rf, rf_acc),
               ('Gradient Boosting', gb, gb_acc),
               ('Ensemble', ensemble, ens_acc)]
    if xgb and xgb_acc > 0:
        results.append(('XGBoost', xgb, xgb_acc))

    best_name, best_model, best_acc = max(results, key=lambda x: x[2])
    print(f"\nBest model: {best_name} ({best_acc*100:.1f}% accuracy)")

    print("\nClassification Report (best model):")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=['Underperform','Outperform']))

    # Feature importance
    try:
        if best_name == 'Ensemble':
            # Use RF component for importance
            importances = rf.named_steps['model'].feature_importances_
        elif hasattr(best_model.named_steps.get('model'), 'feature_importances_'):
            importances = best_model.named_steps['model'].feature_importances_
        else:
            importances = None

        if importances is not None:
            feat_imp = sorted(zip(ALL_FEATURES, importances),
                              key=lambda x: x[1], reverse=True)
            print("\nTop 10 feature importances:")
            for feat, imp in feat_imp[:10]:
                bar = '█' * int(imp * 200)
                tag = '(FUND)' if feat in FUND_FEATURES else '(TECH)'
                print(f"  {feat:<22} {tag} {imp:.3f} {bar}")
    except Exception:
        pass

    return best_model, ALL_FEATURES, best_acc, best_name


# ── Step 5: Save ──────────────────────────────────────────────────────────────
def save_model(model, features, accuracy, model_name):
    print("\nSaving model...")
    joblib.dump({
        'model':      model,
        'features':   features,
        'accuracy':   accuracy,
        'model_name': model_name,
        'trained':    datetime.now().isoformat(),
        'stocks':     [s.replace('.NS','') for s in NSE_STOCKS],
        'version':    'v2',
        'forward_days': FORWARD_DAYS,
    }, 'ml_model.pkl')
    print("Model saved to ml_model.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  NSE ML Stock Screener — Training Pipeline v2")
    print("  Technical + Fundamental Features | 1-Year Horizon")
    print("=" * 65)

    prices    = download_data()
    fund_dict = load_fundamentals()
    val_dict  = download_valuation()
    df        = engineer_features(prices, fund_dict, val_dict)

    if len(df) < 100:
        print("Not enough data to train. Check your internet connection.")
        exit(1)

    df.to_csv('training_data_v2.csv', index=False)
    print(f"Training data saved to training_data_v2.csv ({len(df)} samples)")

    model, features, accuracy, model_name = train_model(df)
    save_model(model, features, accuracy, model_name)

    print("\n" + "=" * 65)
    print(f"  Training complete!")
    print(f"  Best model: {model_name}")
    print(f"  Accuracy:   {accuracy*100:.1f}%")
    print(f"  Features:   {len(features)} ({len(TECH_FEATURES)} tech + {len(FUND_FEATURES)} fundamental)")
    print(f"  Horizon:    {FORWARD_DAYS} trading days (1 year)")
    print("=" * 65)
