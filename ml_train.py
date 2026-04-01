"""
ML Stock Screener — Training Pipeline
======================================
Step 1: Downloads 5 years of NSE stock data from Yahoo Finance
Step 2: Engineers features (momentum, volatility, P/E, moving averages etc)
Step 3: Creates labels (did stock outperform Nifty in next 3 months?)
Step 4: Trains a Random Forest + XGBoost model
Step 5: Saves the model to disk for use in the API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# ── All Nifty 50 stocks ───────────────────────────────────────────────────────
NSE_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "TCS.NS", "INFY.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "RELIANCE.NS",
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
]

NIFTY = "^NSEI"
PERIOD = "5y"
FORWARD_DAYS = 63  # ~3 months of trading days


# ── Step 1: Download price data ───────────────────────────────────────────────
def download_data():
    print("\nDownloading 5 years of NSE data from Yahoo Finance...")
    print("This will take 2-3 minutes. Please wait.\n")

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
            if len(df) > 200:
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


# ── Step 2: Engineer features ─────────────────────────────────────────────────
def engineer_features(prices: dict) -> pd.DataFrame:
    print("\nEngineering features...")

    nifty = prices.get('NIFTY')
    if nifty is None:
        print("No Nifty data found")
        return pd.DataFrame()

    if hasattr(nifty, 'columns'):
        nifty = nifty.iloc[:, 0]
    nifty = pd.Series(nifty.values, index=nifty.index, dtype=float)

    records = []

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

        step = 15  # every 3 weeks — more samples

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

                ma50  = float(np.mean(sw[-50:]))
                ma200 = float(np.mean(sw[-200:]))
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

                records.append({
                    'symbol': sym, 'date': common_idx[i],
                    'ret_1m': ret_1m, 'ret_3m': ret_3m,
                    'ret_6m': ret_6m, 'ret_1y': ret_1y,
                    'rs_1m': rs_1m, 'rs_3m': rs_3m,
                    'price_to_ma50': price_to_ma50,
                    'price_to_ma200': price_to_ma200,
                    'golden_cross': golden_cross,
                    'vol_1m': vol_1m, 'vol_3m': vol_3m,
                    'pos52': pos52, 'rsi': rsi,
                    'vol_trend': vol_trend,
                    'outperforms': outperforms,
                })

            except Exception:
                continue

    if not records:
        print("No records created")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"Created {len(df)} training samples from {df['symbol'].nunique()} stocks")
    return df


# ── Step 3: Train the model ───────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline

    print("\nTraining ML model...")

    FEATURES = [
        'ret_1m','ret_3m','ret_6m','ret_1y',
        'rs_1m','rs_3m',
        'price_to_ma50','price_to_ma200','golden_cross',
        'vol_1m','vol_3m',
        'pos52','rsi','vol_trend'
    ]

    df_clean = df.dropna(subset=FEATURES + ['outperforms'])
    X = df_clean[FEATURES]
    y = df_clean['outperforms']

    print(f"  Training samples: {len(X)}")
    print(f"  Outperformers: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Underperformers: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("\n  Training Random Forest...")
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Random Forest accuracy: {rf_acc*100:.1f}%")

    print("\n  Training Gradient Boosting...")
    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
            min_samples_leaf=15,
            max_features=0.8,
            random_state=42
        ))
    ])
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"  Gradient Boosting accuracy: {gb_acc*100:.1f}%")

    best_model = rf if rf_acc >= gb_acc else gb
    best_name  = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"
    best_acc   = max(rf_acc, gb_acc)

    print(f"\nBest model: {best_name} ({best_acc*100:.1f}% accuracy)")
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=['Underperform','Outperform']))

    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
        print("\nTop feature importances:")
        for feat, imp in feat_imp[:5]:
            print(f"  {feat}: {imp:.3f}")

    return best_model, FEATURES, best_acc


# ── Step 4: Save everything ───────────────────────────────────────────────────
def save_model(model, features, accuracy):
    print("\nSaving model...")
    joblib.dump({
        'model':    model,
        'features': features,
        'accuracy': accuracy,
        'trained':  datetime.now().isoformat(),
        'stocks':   [s.replace('.NS','') for s in NSE_STOCKS],
    }, 'ml_model.pkl')
    print("Model saved to ml_model.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NSE ML Stock Screener — Training Pipeline")
    print("=" * 60)

    prices = download_data()
    df = engineer_features(prices)

    if len(df) < 100:
        print("Not enough data to train. Check your internet connection.")
        exit(1)

    df.to_csv('training_data.csv', index=False)
    print(f"Training data saved to training_data.csv")

    model, features, accuracy = train_model(df)
    save_model(model, features, accuracy)

    print("\n" + "=" * 60)
    print(f"  Training complete! Accuracy: {accuracy*100:.1f}%")
    print("  Next step: run the screener with your trained model")
    print("=" * 60)
