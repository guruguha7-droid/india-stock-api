"""
Screener.in Scraper — Fixed Version
=====================================
Scrapes 10-year fundamental data for Indian stocks.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Symbols that differ between NSE and Screener.in URLs
SCREENER_SYMBOL_MAP = {
    'LTIM':       'LTIMINDTREE',
    'M&M':        'M-M',
    'BAJAJ-AUTO': 'BAJAJ-AUTO',
}


def get_page(symbol: str) -> BeautifulSoup:
    symbol = SCREENER_SYMBOL_MAP.get(symbol, symbol)
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 404:
        url = f"https://www.screener.in/company/{symbol}/"
        r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, 'html.parser')


def parse_num(s: str):
    if not s or s.strip() in ['', '-', '--']:
        return None
    try:
        return float(s.strip().replace(',', '').replace('%', '').replace('₹', ''))
    except:
        return None


def parse_table(soup: BeautifulSoup, section_id: str) -> dict:
    """
    Parse a screener table into dict:
    { 'row_label': [val1, val2, ...], '_years': ['Mar 2014', ...] }
    """
    section = soup.find('section', {'id': section_id})
    if not section:
        return {}
    table = section.find('table')
    if not table:
        return {}

    # Years from header
    years = []
    thead = table.find('thead')
    if thead:
        for th in thead.find_all('th')[1:]:  # skip first empty th
            years.append(th.get_text(strip=True))

    data = {'_years': years}
    tbody = table.find('tbody')
    if tbody:
        for tr in tbody.find_all('tr'):
            cells = tr.find_all('td')
            if not cells:
                continue
            # Clean label — remove + suffix
            label = cells[0].get_text(strip=True).rstrip('+').strip()
            values = [parse_num(td.get_text(strip=True)) for td in cells[1:]]
            if label:
                data[label] = values

    return data


def cagr(values: list, years: int) -> float:
    """Calculate CAGR over given years from end of list."""
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < years + 1:
        return None
    end = clean[-1]
    start = clean[-(years + 1)]
    if start <= 0:
        return None
    return round(((end / start) ** (1 / years) - 1) * 100, 1)


def last(values: list):
    """Get last non-None value."""
    clean = [v for v in values if v is not None]
    return clean[-1] if clean else None


def avg_last_n(values: list, n: int):
    """Average of last n non-None values."""
    clean = [v for v in values if v is not None]
    if len(clean) < n:
        return None
    return round(sum(clean[-n:]) / n, 1)


def trend(values: list, n: int = 5):
    """Latest value minus value n periods ago — positive = improving."""
    clean = [v for v in values if v is not None]
    if len(clean) < n + 1:
        return None
    return round(clean[-1] - clean[-(n + 1)], 1)


def scrape_stock(symbol: str) -> dict:
    try:
        soup = get_page(symbol)

        pl  = parse_table(soup, 'profit-loss')
        bs  = parse_table(soup, 'balance-sheet')
        cf  = parse_table(soup, 'cash-flow')
        rat = parse_table(soup, 'ratios')

        # Shareholding — latest values
        sh_section = soup.find('section', {'id': 'shareholding'})
        holding = {}
        if sh_section:
            table = sh_section.find('table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    for tr in tbody.find_all('tr'):
                        cells = tr.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True).rstrip('+').strip()
                            latest = parse_num(cells[-1].get_text(strip=True))
                            if label and latest is not None:
                                holding[label] = latest

        result = {'symbol': symbol, 'status': 'ok'}

        # ── P&L ──────────────────────────────────────────────────────
        # Banks use 'Revenue' or 'Interest Earned' instead of 'Sales'
        sales   = pl.get('Sales', []) or pl.get('Revenue', []) or pl.get('Interest Earned', [])
        profit  = pl.get('Net Profit', [])
        eps     = pl.get('EPS in Rs', [])
        # Banks use 'Financing Margin %' instead of 'OPM %'
        opm     = pl.get('OPM %', []) or pl.get('Financing Margin %', [])
        div_pay = pl.get('Dividend Payout %', [])

        result['sales_latest_cr']    = last(sales)
        result['sales_cagr_5y']      = cagr(sales, 5)
        result['sales_cagr_10y']     = cagr(sales, 10)
        result['sales_growth_1y']    = cagr(sales, 1)

        result['profit_latest_cr']   = last(profit)
        result['profit_cagr_5y']     = cagr(profit, 5)
        result['profit_cagr_10y']    = cagr(profit, 10)
        result['profit_growth_1y']   = cagr(profit, 1)

        result['eps_latest']         = last(eps)
        result['eps_cagr_5y']        = cagr(eps, 5)
        result['eps_growth_1y']      = cagr(eps, 1)

        result['opm_latest_pct']     = last(opm)
        result['opm_avg_5y']         = avg_last_n(opm, 5)
        result['opm_trend_5y']       = trend(opm, 5)  # + = improving margins

        result['dividend_payout_pct'] = last(div_pay)

        # ── Balance Sheet ─────────────────────────────────────────────
        borrowings = bs.get('Borrowings', [])
        equity     = bs.get('Equity Capital', [])
        reserves   = bs.get('Reserves', [])

        result['debt_latest_cr']     = last(borrowings)
        result['debt_growth_1y']     = cagr(borrowings, 1)

        # Debt reducing = good sign
        clean_debt = [v for v in borrowings if v is not None]
        if len(clean_debt) >= 3:
            result['debt_reducing']  = bool(clean_debt[-1] < clean_debt[-3])

        # Net worth and D/E
        eq_val  = last(equity)
        res_val = last(reserves)
        if eq_val and res_val:
            networth = eq_val + res_val
            result['networth_cr'] = round(networth, 0)
            debt_val = last(borrowings)
            if debt_val is not None and networth > 0:
                result['screener_de'] = round(debt_val / networth, 2)

        # ── Cash Flow ─────────────────────────────────────────────────
        ocf = cf.get('Cash from Operating Activity', [])
        fcf = cf.get('Free Cash Flow', [])

        result['ocf_latest_cr']      = last(ocf)
        result['fcf_latest_cr']      = last(fcf)

        clean_ocf = [v for v in ocf if v is not None]
        clean_fcf = [v for v in fcf if v is not None]

        if len(clean_ocf) >= 3:
            result['ocf_positive_3y'] = bool(all(v > 0 for v in clean_ocf[-3:]))
        if len(clean_fcf) >= 3:
            result['fcf_positive_3y'] = bool(all(v > 0 for v in clean_fcf[-3:]))
        if len(clean_fcf) >= 5:
            result['fcf_cagr_5y']    = cagr(fcf, 5)

        # ── Ratios ────────────────────────────────────────────────────
        # Banks use 'ROE %' instead of 'ROCE %'
        roce = rat.get('ROCE %', []) or rat.get('ROE %', [])
        result['roce_latest_pct']    = last(roce)
        result['roce_avg_5y']        = avg_last_n(roce, 5)
        result['roce_trend_5y']      = trend(roce, 5)  # + = improving

        # ── Shareholding ──────────────────────────────────────────────
        result['promoter_pct']       = holding.get('Promoters', 0.0)
        result['fii_pct']            = holding.get('FIIs')
        result['dii_pct']            = holding.get('DIIs')

        # ── Investment quality score (0-100) ──────────────────────────
        score = 50
        # ROCE — higher is better
        roce_val = result.get('roce_latest_pct')
        if roce_val:
            if roce_val > 25:   score += 15
            elif roce_val > 15: score += 10
            elif roce_val > 10: score += 5
            elif roce_val < 5:  score -= 10

        # Sales growth
        sg = result.get('sales_cagr_5y')
        if sg:
            if sg > 20:   score += 10
            elif sg > 12: score += 7
            elif sg > 5:  score += 3
            elif sg < 0:  score -= 10

        # Profit growth
        pg = result.get('profit_cagr_5y')
        if pg:
            if pg > 20:   score += 10
            elif pg > 12: score += 7
            elif pg > 5:  score += 3
            elif pg < 0:  score -= 10

        # FCF positive
        if result.get('fcf_positive_3y'):  score += 8
        if result.get('ocf_positive_3y'):  score += 5

        # Debt reducing
        if result.get('debt_reducing'):    score += 7

        # Margin trend
        opm_tr  = result.get('opm_trend_5y')
        opm_val = result.get('opm_latest_pct')
        # Skip OPM scoring for banks (negative OPM is normal for banks)
        if opm_tr and opm_val and opm_val > 0:
            if opm_tr > 3:    score += 7
            elif opm_tr > 0:  score += 3
            elif opm_tr < -5: score -= 7

        # Promoter holding
        promo = result.get('promoter_pct')
        if promo:
            if promo > 60:    score += 8
            elif promo > 45:  score += 5
            elif promo < 25:  score -= 8

        result['investment_score'] = max(0, min(100, score))

        # Investment grade
        s = result['investment_score']
        result['investment_grade'] = 'A+' if s>=85 else 'A' if s>=75 else 'B' if s>=60 else 'C' if s>=45 else 'D'

        return result

    except Exception as e:
        return {'symbol': symbol, 'status': 'error', 'error': str(e)}


def scrape_all(symbols: list, delay: float = 2.5) -> pd.DataFrame:
    results = []
    print(f"\n Scraping {len(symbols)} stocks from Screener.in...")
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2d}/{len(symbols)}] {sym}...", end=' ', flush=True)
        d = scrape_stock(sym)
        if d['status'] == 'ok':
            print(f"OK  ROCE:{d.get('roce_latest_pct','?')}%  "
                  f"SalesCGR5Y:{d.get('sales_cagr_5y','?')}%  "
                  f"Promoter:{d.get('promoter_pct','?')}%  "
                  f"Score:{d.get('investment_score','?')}")
        else:
            print(f"FAIL  {d.get('error','unknown')}")
        results.append(d)
        if i < len(symbols):
            time.sleep(delay)
    return pd.DataFrame(results)


if __name__ == "__main__":
    from scraper import NSE_STOCKS

    print("="*60)
    print("  Screener.in Scraper — Full Run (79 stocks)")
    print("="*60)
    print("  Estimated time: ~4 minutes (2.5s delay per stock)")
    print("  Being respectful to Screener.in servers\n")

    df = scrape_all(NSE_STOCKS, delay=2.5)

    # Summary
    ok = df[df['status']=='ok']
    print(f"\n{'='*60}")
    print(f"  Scraped: {len(ok)}/{len(df)} stocks successfully")
    print(f"\n  Top 10 by Investment Score:")
    if 'investment_score' in ok.columns:
        top = ok.nlargest(10, 'investment_score')[
            ['symbol','investment_score','investment_grade',
             'roce_latest_pct','sales_cagr_5y','profit_cagr_5y',
             'promoter_pct','fcf_positive_3y']]
        print(top.to_string(index=False))

    df.to_csv('screener_fundamentals.csv', index=False)
    print(f"\n  Full data saved to screener_fundamentals.csv")
