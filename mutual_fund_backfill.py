"""
Mutual Fund Backfill — Historical NAV ingestion from mfapi.in
==============================================================

mfapi.in returns the full NAV history for a scheme in one JSON call.
We use it for one-time backfill; AMFI itself is used for daily forward updates.

Key design:
- Idempotent: re-runs only fetch what's missing per scheme (uses scheme_backfill_meta)
- Polite rate limiting (4 req/sec default; mfapi.in is free and shared)
- Resume-safe: crashes mid-run don't lose progress (per-scheme commit)
- Configurable max_years: re-run with higher value later to extend history
"""

import time
import logging
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import psycopg2.extras

from mutual_fund_data import db_cursor

logger = logging.getLogger('graham.mf_backfill')

MFAPI_URL = "https://api.mfapi.in/mf/{scheme_code}"


def _parse_mfapi_date(d):
    """mfapi.in returns dates as 'dd-mm-yyyy'. Convert to date object."""
    try:
        return datetime.strptime(d, '%d-%m-%Y').date()
    except (ValueError, TypeError):
        return None


def fetch_scheme_history(scheme_code, timeout=15):
    """
    Fetch full NAV history for one scheme from mfapi.in.
    Returns list of (nav_date, nav_value) tuples, sorted newest first.
    """
    url = MFAPI_URL.format(scheme_code=scheme_code)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json() or {}
    entries = data.get('data') or []
    if not entries:
        return []

    parsed = []
    for entry in entries:
        d = _parse_mfapi_date(entry.get('date'))
        if not d:
            continue
        try:
            nav = float(entry.get('nav', '').replace(',', ''))
        except (ValueError, TypeError):
            continue
        if nav <= 0:
            continue
        parsed.append((d, nav))

    parsed.sort(key=lambda x: x[0], reverse=True)
    return parsed


def get_schemes_to_backfill(max_years, force=False):
    """
    Determine which schemes need (more) backfill data.

    Returns: list of (scheme_code, current_oldest_nav_date) tuples.
    If force=True, returns every scheme regardless of current state.
    """
    target_cutoff = date.today() - timedelta(days=max_years * 365)

    with db_cursor() as cur:
        if force:
            cur.execute("""
                SELECT s.scheme_code, m.oldest_nav_date
                FROM schemes s
                LEFT JOIN scheme_backfill_meta m USING (scheme_code)
                ORDER BY s.scheme_code
            """)
        else:
            cur.execute("""
                SELECT s.scheme_code, m.oldest_nav_date
                FROM schemes s
                LEFT JOIN scheme_backfill_meta m USING (scheme_code)
                WHERE m.oldest_nav_date IS NULL
                   OR m.oldest_nav_date > %s
                ORDER BY s.scheme_code
            """, (target_cutoff,))
        return [(r['scheme_code'], r['oldest_nav_date']) for r in cur.fetchall()]


def upsert_navs_for_scheme(scheme_code, nav_pairs, cutoff_date):
    """
    Insert NAV history for one scheme. Filters to dates >= cutoff_date.
    Returns: count of NAVs actually inserted (new rows only).
    """
    filtered = [(d, v) for d, v in nav_pairs if d >= cutoff_date]
    if not filtered:
        return 0

    records = [(scheme_code, d, v) for d, v in filtered]

    with db_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO nav_history (scheme_code, nav_date, nav_value)
            VALUES %s
            ON CONFLICT (scheme_code, nav_date) DO NOTHING
            """,
            records,
            page_size=1000,
        )
        return cur.rowcount


def update_backfill_meta(scheme_code, status='ok', error=None):
    """Update scheme_backfill_meta with current state of nav_history for this scheme."""
    with db_cursor() as cur:
        cur.execute("""
            INSERT INTO scheme_backfill_meta
              (scheme_code, oldest_nav_date, newest_nav_date, nav_count,
               last_backfill_at, last_backfill_status, last_error)
            SELECT
              %s,
              (SELECT MIN(nav_date) FROM nav_history WHERE scheme_code = %s),
              (SELECT MAX(nav_date) FROM nav_history WHERE scheme_code = %s),
              (SELECT COUNT(*) FROM nav_history WHERE scheme_code = %s),
              NOW(),
              %s,
              %s
            ON CONFLICT (scheme_code) DO UPDATE SET
              oldest_nav_date      = EXCLUDED.oldest_nav_date,
              newest_nav_date      = EXCLUDED.newest_nav_date,
              nav_count            = EXCLUDED.nav_count,
              last_backfill_at     = EXCLUDED.last_backfill_at,
              last_backfill_status = EXCLUDED.last_backfill_status,
              last_error           = EXCLUDED.last_error
        """, (scheme_code, scheme_code, scheme_code, scheme_code, status, error))


def get_backfill_status():
    """Return current backfill progress for monitoring."""
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM schemes")
        total_schemes = cur.fetchone()['n']

        cur.execute("SELECT COUNT(*) AS n FROM scheme_backfill_meta")
        backfilled = cur.fetchone()['n']

        cur.execute("SELECT COUNT(*) AS n FROM scheme_backfill_meta WHERE last_backfill_status = 'ok'")
        successful = cur.fetchone()['n']

        cur.execute("SELECT COUNT(*) AS n FROM nav_history")
        total_navs = cur.fetchone()['n']

        cur.execute("SELECT MIN(oldest_nav_date) AS d FROM scheme_backfill_meta WHERE oldest_nav_date IS NOT NULL")
        oldest = cur.fetchone()['d']

        cur.execute("""
            SELECT scheme_code, last_error, last_backfill_at
            FROM scheme_backfill_meta
            WHERE last_backfill_status = 'error'
            ORDER BY last_backfill_at DESC LIMIT 5
        """)
        recent_errors = [
            {'scheme_code': r['scheme_code'],
             'error': r['last_error'][:200] if r['last_error'] else None,
             'at': str(r['last_backfill_at'])}
            for r in cur.fetchall()
        ]

    return {
        'total_schemes':      total_schemes,
        'schemes_backfilled': backfilled,
        'schemes_successful': successful,
        'schemes_pending':    total_schemes - backfilled,
        'percent_complete':   round(backfilled * 100.0 / total_schemes, 1) if total_schemes else 0,
        'total_nav_rows':     total_navs,
        'oldest_nav_in_db':   str(oldest) if oldest else None,
        'recent_errors':      recent_errors,
    }


def _process_one_scheme(scheme_code, cutoff_date):
    """Worker: fetch one scheme's history and store it. Returns (code, status, navs_added)."""
    try:
        nav_pairs = fetch_scheme_history(scheme_code)
        if not nav_pairs:
            update_backfill_meta(scheme_code, status='no_data', error='mfapi.in returned no NAVs')
            return (scheme_code, 'no_data', 0)
        navs_added = upsert_navs_for_scheme(scheme_code, nav_pairs, cutoff_date)
        update_backfill_meta(scheme_code, status='ok')
        return (scheme_code, 'ok', navs_added)
    except Exception as e:
        try:
            update_backfill_meta(scheme_code, status='error', error=str(e)[:500])
        except Exception:
            pass
        return (scheme_code, 'error', 0)


def run_backfill(max_years=10, rate_limit_qps=4, force=False, max_workers=8,
                 progress_every=100, limit=None):
    """
    Main backfill driver.

    Args:
      max_years:       how far back to fetch (e.g. 10)
      rate_limit_qps:  max requests per second to mfapi.in (be polite)
      force:           re-process all schemes even if already covered
      max_workers:     parallelism (keep <= 10 to be polite to mfapi.in)
      progress_every:  print progress every N schemes
      limit:           cap on number of schemes to process (None = all, for testing)
    """
    started = time.time()
    target_cutoff = date.today() - timedelta(days=max_years * 365)

    schemes = get_schemes_to_backfill(max_years, force=force)
    if limit:
        schemes = schemes[:limit]

    total = len(schemes)
    logger.info(f"Backfill: starting for {total} schemes | max_years={max_years} | cutoff={target_cutoff}")
    print(f"[backfill] starting | {total} schemes | max_years={max_years} | cutoff={target_cutoff}")

    if total == 0:
        print("[backfill] nothing to do (use force=True to re-process all)")
        return {'status': 'ok', 'schemes_processed': 0, 'elapsed_s': 0}

    stats = {'ok': 0, 'no_data': 0, 'error': 0, 'total_navs_added': 0}
    last_print = time.time()
    interval = 1.0 / rate_limit_qps

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for scheme_code, _ in schemes:
            futures[executor.submit(_process_one_scheme, scheme_code, target_cutoff)] = scheme_code
            time.sleep(interval)

        for future in as_completed(futures):
            code, status, n_navs = future.result()
            stats[status] = stats.get(status, 0) + 1
            stats['total_navs_added'] += n_navs

            done = stats['ok'] + stats['no_data'] + stats['error']
            if done % progress_every == 0 or time.time() - last_print > 30:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total - done) / rate if rate > 0 else 0
                print(f"[backfill] {done}/{total} ({100*done/total:.1f}%) | "
                      f"ok={stats['ok']} no_data={stats['no_data']} err={stats['error']} | "
                      f"navs+={stats['total_navs_added']} | "
                      f"rate={rate:.1f}/s | eta={eta_s/60:.1f}min")
                last_print = time.time()

    elapsed = time.time() - started
    print(f"[backfill] DONE in {elapsed/60:.1f} min | "
          f"ok={stats['ok']} no_data={stats['no_data']} err={stats['error']} | "
          f"navs+={stats['total_navs_added']}")

    return {
        'status':            'ok',
        'schemes_processed': total,
        'elapsed_s':         round(elapsed, 1),
        **stats,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill mutual fund NAV history')
    parser.add_argument('--max-years', type=int, default=10, help='How many years back to fetch')
    parser.add_argument('--qps', type=float, default=4, help='Max requests per second')
    parser.add_argument('--force', action='store_true', help='Re-process all schemes')
    parser.add_argument('--limit', type=int, default=None, help='Limit schemes (for testing)')
    parser.add_argument('--workers', type=int, default=8, help='Concurrent workers')
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(dotenv_path='.env')

    from mutual_fund_data import init_schema
    init_schema()

    result = run_backfill(
        max_years=args.max_years,
        rate_limit_qps=args.qps,
        force=args.force,
        max_workers=args.workers,
        limit=args.limit,
    )
    print(f"\nFinal: {result}")
