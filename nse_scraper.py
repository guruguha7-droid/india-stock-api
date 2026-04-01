"""
NSE Real-Time Scraper
======================
Fetches live stock prices directly from NSE India website.
Data is real-time (seconds delay) — much better than yfinance's 15 min delay.

Requires browser-like session management to avoid NSE blocking.
Session is refreshed every 25 minutes automatically.
"""

import requests
import time
import threading

# ── Session management ────────────────────────────────────────────────────────
_session = None
_session_ts = 0
SESSION_TTL = 1500  # refresh every 25 minutes

_session_lock = threading.Lock()

NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Referer': 'https://www.nseindia.com/',
    'Connection': 'keep-alive',
    'X-Requested-With': 'XMLHttpRequest',
}


def get_session() -> requests.Session:
    """Get or refresh NSE session."""
    global _session, _session_ts
    now = time.time()

    with _session_lock:
        if _session is None or (now - _session_ts) > SESSION_TTL:
            try:
                s = requests.Session()
                s.headers.update(NSE_HEADERS)
                # Visit market data page to get full cookie set
                s.get('https://www.nseindia.com/market-data/live-equity-market',
                      timeout=10)
                time.sleep(1)  # let cookies settle before API calls
                _session    = s
                _session_ts = now
                print("  NSE session refreshed")
            except Exception as e:
                print(f"  NSE session error: {e}")
                if _session is None:
                    _session = requests.Session()
                    _session.headers.update(NSE_HEADERS)

    return _session


def fetch_nse(url: str, retries: int = 2) -> dict:
    """Fetch NSE API endpoint with session + retry."""
    global _session_ts
    for attempt in range(retries + 1):
        try:
            s = get_session()
            r = s.get(url, timeout=8)
            if r.status_code in (401, 403):
                # Session expired — force refresh
                _session_ts = 0
                s = get_session()
                r = s.get(url, timeout=8)
            if r.status_code == 200:
                return r.json()
            if attempt < retries:
                time.sleep(1)
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                raise e
    return {}


def get_quote(symbol: str) -> dict:
    """
    Get real-time quote for a single NSE stock.
    Returns standardised dict with all price fields.
    """
    try:
        data = fetch_nse(
            f'https://www.nseindia.com/api/quote-equity?symbol={symbol}'
        )
        if not data:
            return {'symbol': symbol, 'error': 'No data from NSE'}

        pi   = data.get('priceInfo', {})
        meta = data.get('metadata', {})
        info = data.get('securityInfo', {})

        # Price data
        price      = pi.get('lastPrice')
        change     = pi.get('change')
        change_pct = pi.get('pChange')
        prev_close = pi.get('previousClose')
        open_price = pi.get('open')
        vwap       = pi.get('vwap')

        # Intraday high/low
        intra = pi.get('intraDayHighLow', {})
        high  = intra.get('max')
        low   = intra.get('min')

        # 52-week high/low
        week        = pi.get('weekHighLow', {})
        w_high      = week.get('max')
        w_low       = week.get('min')
        w_high_date = week.get('maxDate', '')
        w_low_date  = week.get('minDate', '')

        # Position in 52W range (0=at low, 100=at high)
        pos52 = None
        if price and w_high and w_low and w_high != w_low:
            pos52 = round((price - w_low) / (w_high - w_low) * 100, 1)

        # Meta info
        company_name = meta.get('companyName', symbol)
        industry     = meta.get('industry', '')
        isin         = meta.get('isin', '')
        series       = meta.get('series', 'EQ')

        # Security info
        face_value  = info.get('faceValue')
        issued_size = info.get('issuedSize')

        # Market cap (approximate)
        market_cap = None
        if price and issued_size:
            try:
                mc = float(price) * float(issued_size) / 1e7  # in crores
                if mc >= 1e5:
                    market_cap = f"₹{mc/1e5:.2f}L Cr"
                elif mc >= 1e3:
                    market_cap = f"₹{mc/1e3:.2f}T Cr"
                else:
                    market_cap = f"₹{mc:.0f} Cr"
            except Exception:
                pass

        return {
            'symbol':       symbol,
            'company_name': company_name,
            'industry':     industry,
            'series':       series,
            'isin':         isin,
            # Price
            'price':        round(float(price), 2) if price else None,
            'change':       round(float(change), 2) if change else None,
            'change_pct':   round(float(change_pct), 2) if change_pct else None,
            'prev_close':   round(float(prev_close), 2) if prev_close else None,
            'open':         round(float(open_price), 2) if open_price else None,
            'high':         round(float(high), 2) if high else None,
            'low':          round(float(low), 2) if low else None,
            'vwap':         round(float(vwap), 2) if vwap else None,
            # 52W
            'week52_high':      round(float(w_high), 2) if w_high else None,
            'week52_low':       round(float(w_low), 2) if w_low else None,
            'week52_high_date': w_high_date,
            'week52_low_date':  w_low_date,
            'pos52_pct':        pos52,
            # Market
            'market_cap':  market_cap,
            'face_value':  face_value,
            'source':      'NSE',
            'delay':       'real-time',
        }

    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def get_indices() -> dict:
    """Get live NSE indices — Nifty 50, Bank Nifty, India VIX, Sensex, USD/INR."""
    try:
        data = fetch_nse('https://www.nseindia.com/api/allIndices')
        indices = {}

        for item in data.get('data', []):
            name = item.get('index', '')
            if name in ['NIFTY 50', 'NIFTY BANK', 'INDIA VIX']:
                indices[name] = {
                    'price':      item.get('last'),
                    'change':     item.get('change'),
                    'change_pct': item.get('percentChange'),
                }

        # Sensex via yfinance ^BSESN (more reliable than Stooq)
        try:
            import yfinance as _yf
            bse = _yf.download("^BSESN", period="2d", interval="1d",
                               auto_adjust=True, progress=False)
            if bse is not None and len(bse) >= 1:
                if hasattr(bse.columns, 'levels'):
                    bse.columns = bse.columns.get_level_values(0)
                close_vals = bse['Close'].squeeze()
                price = float(close_vals.iloc[-1])
                prev  = float(close_vals.iloc[-2]) if len(close_vals) >= 2 else price
                if price > 0:
                    chg  = round(price - prev, 2)
                    chgp = round((chg / prev) * 100, 2) if prev > 0 else 0.0
                    indices['SENSEX'] = {
                        'price':      round(price, 2),
                        'change':     chg,
                        'change_pct': chgp,
                    }
        except Exception:
            pass

        # USD/INR via frankfurter free API
        try:
            r = requests.get(
                'https://api.frankfurter.app/latest?from=USD&to=INR',
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=5
            )
            if r.status_code == 200:
                j = r.json()
                inr = j.get('rates', {}).get('INR')
                if inr:
                    indices['USD_INR'] = {
                        'price':      round(float(inr), 2),
                        'change':     None,
                        'change_pct': None,
                    }
        except Exception:
            pass

        return indices

    except Exception as e:
        return {'error': str(e)}


def get_multiple_quotes(symbols: list) -> dict:
    """
    Get real-time quotes for multiple stocks efficiently.
    Uses threading to fetch in parallel.
    """
    results = {}
    lock = threading.Lock()

    def fetch_one(sym):
        q = get_quote(sym)
        with lock:
            results[sym] = q
        time.sleep(0.1)

    threads = []
    for sym in symbols:
        t = threading.Thread(target=fetch_one, args=(sym,))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # stagger starts to avoid hammering NSE

    for t in threads:
        t.join(timeout=10)

    return results


if __name__ == "__main__":
    print("="*50)
    print("  NSE Real-Time Scraper Test")
    print("="*50)

    test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'HAL', 'MAZDOCK']

    print("\nFetching real-time quotes...")
    for sym in test_stocks:
        q = get_quote(sym)
        if 'error' not in q:
            chg = q['change_pct']
            arrow = '+' if chg and chg >= 0 else '-'
            print(f"  {sym:<15} Rs.{q['price']:<10} {arrow}{abs(chg):.2f}%  "
                  f"52W: Rs.{q['week52_low']} - Rs.{q['week52_high']}")
        else:
            print(f"  {sym:<15} ERROR: {q['error']}")
        time.sleep(0.5)

    print("\nFetching indices...")
    idx = get_indices()
    for name, d in idx.items():
        if 'price' in d:
            chg = d['change_pct'] or 0
            print(f"  {name}: {d['price']} ({chg:+.2f}%)")
