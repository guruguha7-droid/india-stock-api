# 🇮🇳 India Stock Scraper + API

Scrapes **live NSE stock data from Yahoo Finance** using Selenium (real browser automation), exactly like the tutorial. Then serves it as a JSON API with CORS so your Claude.ai widget can call it directly.

---

## 📦 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🌐 2. Install ChromeDriver

ChromeDriver must match your installed Chrome version.

**Option A — Auto (recommended):**
```bash
pip install webdriver-manager
```
Then change `scraper.py` line:
```python
# Replace:
driver = webdriver.Chrome(options=options)

# With:
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
```

**Option B — Manual:**
1. Check your Chrome version: `chrome://settings/help`
2. Download matching driver: https://chromedriver.chromium.org/downloads
3. Set path in `scraper.py`:
```python
service = Service("/path/to/chromedriver")
driver = webdriver.Chrome(service=service, options=options)
```

---

## 🔍 3. Run the scraper directly

```bash
python scraper.py
```

This opens Chrome (headless), visits each Yahoo Finance page, extracts:
- Live price
- Change + Change %
- Previous close, Open
- Day range (High/Low)
- 52-week range
- Volume
- Market Cap
- P/E Ratio
- EPS
- Dividend Yield

Saves to `nse_stocks_TIMESTAMP.csv` and `.json`

---

## 🚀 4. Run the API server

```bash
python api_server.py
```

Server starts at **http://localhost:5000**

### Endpoints:

| Endpoint | Description |
|---|---|
| `GET /quote?symbol=RELIANCE` | Single stock |
| `GET /quotes?symbols=RELIANCE,TCS,INFY` | Multiple stocks |
| `GET /watchlist` | All 20 default Nifty stocks |
| `GET /indices` | NIFTY 50, SENSEX, BANK NIFTY |
| `GET /health` | Health check |

### Example response:
```json
{
  "status": "ok",
  "data": {
    "symbol": "RELIANCE",
    "price": "1,452.30",
    "change": "+18.50",
    "change_pct": "(+1.29%)",
    "market_cap": "19.62T",
    "pe_ratio": "27.4",
    "eps": "53.1",
    "week52_high": "1,608.95",
    "week52_low": "1,175.00"
  }
}
```

---

## ☁️ 5. Deploy online (so the Claude widget can call it)

### Railway (free tier, easiest):
1. Push this folder to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add a `Procfile`:
   ```
   web: python api_server.py
   ```
4. Railway gives you a public URL like `https://your-app.railway.app`

### Render (free tier):
1. Push to GitHub
2. Go to https://render.com → New Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `python api_server.py`

> **Note for cloud deploy:** Chrome won't be pre-installed. Add this to your build:
> ```bash
> apt-get install -y chromium-browser
> ```
> And use `chromium-browser` instead of `chrome` in your driver options.

---

## 🔗 6. Connect to Claude widget

Once deployed, share your API URL. The widget will call:
```
https://your-api.railway.app/quotes?symbols=RELIANCE,TCS,HDFCBANK
```

CORS is already enabled — no extra config needed.

---

## ⚙️ Customize which stocks to scrape

Edit `NSE_STOCKS` in `scraper.py`:
```python
NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK",
    # add any NSE ticker here...
]
```

---

## 📝 How it works (tutorial steps)

1. **Selenium opens a real Chrome browser** (headless = invisible)
2. **Navigates to** `https://finance.yahoo.com/quote/RELIANCE.NS`
3. **Waits** for the price element to appear (dynamic JS content)
4. **Extracts** price via CSS selector: `fin-streamer[data-field="regularMarketPrice"]`
5. **Loops** through all stocks in the list
6. **Stores** everything in a pandas DataFrame → CSV/JSON
7. **Flask API** serves the data with CORS headers
