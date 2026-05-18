"""
Mutual Fund Data Module — AMFI NAV scraper + Postgres storage
==============================================================

This module is responsible for:
1. Fetching daily NAV data from AMFI (https://www.amfiindia.com/spages/NAVAll.txt)
2. Parsing the pipe-delimited format
3. Storing schemes (master data) and NAV history (time-series) in Postgres
4. Providing query helpers for the API endpoints

Database schema:
- schemes:      master data per scheme (code, name, AMC, category, ISIN)
- nav_history:  daily NAV values keyed by (scheme_code, nav_date)
"""

import os
import logging
import re
import time
from datetime import date, datetime, timedelta
from contextlib import contextmanager

import requests
import psycopg2
import psycopg2.extras

logger = logging.getLogger('graham.mutual_funds')

# AMFI's daily NAV file — updated every business day after market close
AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

# ── Database connection ───────────────────────────────────────────────────────

def _get_db_url():
    """Read DATABASE_URL from environment. Loads .env file if available locally."""
    url = os.environ.get('DATABASE_URL')
    if not url:
        # Try loading .env (for local dev)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            url = os.environ.get('DATABASE_URL')
        except ImportError:
            pass
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Add it to environment variables (Render) "
            "or to a .env file (locally)."
        )

    # Strip params psycopg2 can't parse (Neon adds channel_binding=require)
    # sslmode=require is already in the URL, so TLS is still enforced.
    if '?' in url:
        base, params = url.split('?', 1)
        kept = []
        for kv in params.split('&'):
            key = kv.split('=', 1)[0].lower()
            if key in ('channel_binding',):
                continue
            kept.append(kv)
        url = base + ('?' + '&'.join(kept) if kept else '')

    return url


@contextmanager
def db_cursor():
    """Context manager that yields a cursor. Commits on success, rolls back on error."""
    conn = psycopg2.connect(_get_db_url(), connect_timeout=10)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialization ─────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schemes (
    scheme_code     VARCHAR(20) PRIMARY KEY,
    scheme_name     TEXT NOT NULL,
    amc_name        TEXT,
    category        TEXT,
    sub_category    TEXT,
    isin_growth     VARCHAR(20),
    isin_div_reinv  VARCHAR(20),
    first_seen      TIMESTAMP DEFAULT NOW(),
    last_seen       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schemes_amc ON schemes(amc_name);
CREATE INDEX IF NOT EXISTS idx_schemes_category ON schemes(category);

CREATE TABLE IF NOT EXISTS nav_history (
    scheme_code     VARCHAR(20) NOT NULL REFERENCES schemes(scheme_code),
    nav_date        DATE NOT NULL,
    nav_value       NUMERIC(14, 4) NOT NULL,
    repurchase_price NUMERIC(14, 4),
    sale_price      NUMERIC(14, 4),
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (scheme_code, nav_date)
);

CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history(nav_date DESC);
CREATE INDEX IF NOT EXISTS idx_nav_scheme_date ON nav_history(scheme_code, nav_date DESC);

CREATE TABLE IF NOT EXISTS fetch_log (
    id              SERIAL PRIMARY KEY,
    fetched_at      TIMESTAMP DEFAULT NOW(),
    nav_date        DATE,
    schemes_added   INTEGER,
    schemes_updated INTEGER,
    navs_added      INTEGER,
    status          VARCHAR(20),
    error           TEXT
);
"""


def init_schema():
    """Initialize database schema. Safe to call multiple times — uses IF NOT EXISTS."""
    with db_cursor() as cur:
        cur.execute(SCHEMA_DDL)
    logger.info("mutual_fund_data: schema initialized")


# ── AMFI fetch + parse ────────────────────────────────────────────────────────

def fetch_amfi_raw(timeout=30):
    """Fetch the raw NAV text from AMFI."""
    resp = requests.get(AMFI_NAV_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_amfi(text):
    """
    Parse AMFI's pipe-delimited format.

    The file has this structure:
      Open Ended Schemes ( Equity Scheme - Large Cap Fund )
      <AMC Name>
      Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

      120503;INF209K01157;INF209K01165;Aditya Birla Sun Life Equity Fund - GROWTH;1234.5678;15-May-2026

    Returns: list of dicts with parsed scheme + NAV data.
    """
    current_category = None       # e.g. "Equity Scheme - Large Cap Fund"
    current_sub_cat = None        # broader bucket like "Open Ended"
    current_amc = None
    rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Data line: scheme code starts with a digit, has 6 fields separated by ';'
        # Format: code;isin1;isin2;name;nav;date
        if ';' in line:
            parts = [p.strip() for p in line.split(';')]
            if len(parts) >= 6 and parts[0].isdigit():
                try:
                    nav_date = datetime.strptime(parts[5], '%d-%b-%Y').date()
                    nav_value = parts[4].replace(',', '')
                    if nav_value in ('N.A.', 'NA', '', 'B.C.'):
                        continue  # NAV not available
                    # Skip dead/discontinued schemes (no NAV in 6+ months)
                    if (date.today() - nav_date).days > 180:
                        continue
                    rows.append({
                        'scheme_code':    parts[0],
                        'isin_growth':    parts[1] if parts[1] not in ('-', '') else None,
                        'isin_div_reinv': parts[2] if parts[2] not in ('-', '') else None,
                        'scheme_name':    parts[3],
                        'nav_value':      float(nav_value),
                        'nav_date':       nav_date,
                        'category':       current_sub_cat,
                        'sub_category':   current_category,
                        'amc_name':       current_amc,
                    })
                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipping unparseable line: {line[:80]} ({e})")
                    continue
            elif parts[0].lower() == 'scheme code':
                continue  # column header row, skip
            continue

        # Non-pipe line: either category header or AMC name
        # AMFI format: "Open Ended Schemes(Debt Scheme - Banking and PSU Fund)"
        # (no space between "Schemes" and "(")
        cat_match = re.match(r'(Open Ended|Close Ended|Interval)\s*Schemes?\s*\(\s*(.+?)\s*\)', line, re.IGNORECASE)
        if cat_match:
            current_sub_cat = cat_match.group(1).title() + ' Schemes'
            current_category = cat_match.group(2).strip()
            continue

        # Otherwise treat as AMC name (mutual fund company)
        if line and not line.startswith('Scheme Code'):
            current_amc = line

    return rows


# ── Storage operations ───────────────────────────────────────────────────────

def upsert_schemes_and_navs(parsed_rows):
    """
    Insert/update scheme master + NAV history from parsed rows.

    Uses ON CONFLICT for idempotent upserts — safe to run multiple times for same date.

    Returns: dict with counts (schemes_added, schemes_updated, navs_added).
    """
    if not parsed_rows:
        return {'schemes_added': 0, 'schemes_updated': 0, 'navs_added': 0}

    schemes_added = 0
    schemes_updated = 0
    navs_added = 0

    with db_cursor() as cur:
        scheme_records = [
            (
                r['scheme_code'], r['scheme_name'], r['amc_name'],
                r['category'], r['sub_category'],
                r['isin_growth'], r['isin_div_reinv'],
            )
            for r in parsed_rows
        ]

        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO schemes
              (scheme_code, scheme_name, amc_name, category, sub_category,
               isin_growth, isin_div_reinv)
            VALUES %s
            ON CONFLICT (scheme_code) DO UPDATE SET
              scheme_name    = EXCLUDED.scheme_name,
              amc_name       = COALESCE(EXCLUDED.amc_name, schemes.amc_name),
              category       = COALESCE(EXCLUDED.category, schemes.category),
              sub_category   = COALESCE(EXCLUDED.sub_category, schemes.sub_category),
              isin_growth    = COALESCE(EXCLUDED.isin_growth, schemes.isin_growth),
              isin_div_reinv = COALESCE(EXCLUDED.isin_div_reinv, schemes.isin_div_reinv),
              last_seen      = NOW()
            RETURNING scheme_code, (xmax = 0) AS inserted
            """,
            scheme_records,
            page_size=500,
        )
        for row in cur.fetchall():
            if row['inserted']:
                schemes_added += 1
            else:
                schemes_updated += 1

        nav_records = [
            (r['scheme_code'], r['nav_date'], r['nav_value'])
            for r in parsed_rows
        ]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO nav_history (scheme_code, nav_date, nav_value)
            VALUES %s
            ON CONFLICT (scheme_code, nav_date) DO NOTHING
            """,
            nav_records,
            page_size=1000,
        )
        navs_added = cur.rowcount

    return {
        'schemes_added':   schemes_added,
        'schemes_updated': schemes_updated,
        'navs_added':      navs_added,
    }


def log_fetch(nav_date, counts, status='ok', error=None):
    """Record a fetch attempt in fetch_log for monitoring."""
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO fetch_log
              (nav_date, schemes_added, schemes_updated, navs_added, status, error)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (nav_date, counts.get('schemes_added', 0),
             counts.get('schemes_updated', 0), counts.get('navs_added', 0),
             status, error),
        )


# ── Public API: the main fetch+store function ─────────────────────────────────

def refresh_amfi_data():
    """
    Fetch latest NAV data from AMFI and persist to DB.

    Returns: dict summarizing what happened.
    Idempotent: if called multiple times for same day, won't duplicate NAVs.
    """
    started = time.time()
    try:
        logger.info("mutual_fund_data: fetching AMFI NAV file")
        raw = fetch_amfi_raw()
        rows = parse_amfi(raw)

        if not rows:
            log_fetch(None, {}, status='error', error='No rows parsed from AMFI')
            return {'status': 'error', 'error': 'AMFI returned no parseable rows'}

        nav_date = rows[0]['nav_date']
        counts = upsert_schemes_and_navs(rows)
        log_fetch(nav_date, counts, status='ok')

        elapsed = round(time.time() - started, 1)
        logger.info(
            f"mutual_fund_data: refreshed in {elapsed}s | "
            f"date={nav_date} | parsed={len(rows)} | "
            f"new_schemes={counts['schemes_added']} | "
            f"updated_schemes={counts['schemes_updated']} | "
            f"new_navs={counts['navs_added']}"
        )

        return {
            'status':    'ok',
            'elapsed_s': elapsed,
            'nav_date':  str(nav_date),
            'parsed':    len(rows),
            **counts,
        }
    except Exception as e:
        logger.exception("mutual_fund_data: refresh failed")
        try:
            log_fetch(None, {}, status='error', error=str(e)[:500])
        except Exception:
            pass
        return {'status': 'error', 'error': str(e)}


# ── Query helpers (used by API endpoints) ─────────────────────────────────────

def list_schemes(limit=100, offset=0, search=None, amc=None, category=None):
    """List schemes with optional filters."""
    sql = "SELECT * FROM schemes WHERE 1=1"
    params = []
    if search:
        sql += " AND scheme_name ILIKE %s"
        params.append(f'%{search}%')
    if amc:
        sql += " AND amc_name ILIKE %s"
        params.append(f'%{amc}%')
    if category:
        sql += " AND (category ILIKE %s OR sub_category ILIKE %s)"
        params.extend([f'%{category}%', f'%{category}%'])
    sql += " ORDER BY scheme_name LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with db_cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def get_scheme(scheme_code):
    """Return one scheme's master record + latest NAV."""
    with db_cursor() as cur:
        cur.execute("SELECT * FROM schemes WHERE scheme_code = %s", (scheme_code,))
        scheme = cur.fetchone()
        if not scheme:
            return None
        cur.execute(
            "SELECT nav_value, nav_date FROM nav_history "
            "WHERE scheme_code = %s ORDER BY nav_date DESC LIMIT 1",
            (scheme_code,)
        )
        latest = cur.fetchone()
        result = dict(scheme)
        if latest:
            result['latest_nav']      = float(latest['nav_value'])
            result['latest_nav_date'] = str(latest['nav_date'])
        return result


def get_nav_history(scheme_code, days=365):
    """Return NAV history for a scheme over last N days."""
    cutoff = date.today() - timedelta(days=days)
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT nav_date, nav_value FROM nav_history
            WHERE scheme_code = %s AND nav_date >= %s
            ORDER BY nav_date ASC
            """,
            (scheme_code, cutoff)
        )
        return [
            {'date': str(r['nav_date']), 'nav': float(r['nav_value'])}
            for r in cur.fetchall()
        ]


def health_check():
    """Return DB connectivity + data freshness stats."""
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM schemes")
        scheme_count = cur.fetchone()['n']

        cur.execute("SELECT COUNT(*) AS n FROM nav_history")
        nav_count = cur.fetchone()['n']

        cur.execute("SELECT MAX(nav_date) AS d FROM nav_history")
        latest_nav_date = cur.fetchone()['d']

        cur.execute("SELECT * FROM fetch_log ORDER BY fetched_at DESC LIMIT 5")
        recent_fetches = [dict(r) for r in cur.fetchall()]

    return {
        'scheme_count':    scheme_count,
        'nav_count':       nav_count,
        'latest_nav_date': str(latest_nav_date) if latest_nav_date else None,
        'recent_fetches':  [
            {
                'fetched_at':      str(r['fetched_at']),
                'nav_date':        str(r['nav_date']) if r['nav_date'] else None,
                'schemes_added':   r['schemes_added'],
                'schemes_updated': r['schemes_updated'],
                'navs_added':      r['navs_added'],
                'status':          r['status'],
                'error':           r['error'],
            }
            for r in recent_fetches
        ],
    }
