"""
Mutual Fund Analytics — Phase 2 of Graham platform.

Computes returns (CAGR), risk (volatility, drawdown), and risk-adjusted
metrics (Sharpe) from stored NAV history. Pure-read; no DB writes.

Design choices:
- Point-to-point CAGR for headline 1Y/3Y/5Y returns (industry standard).
- Rolling-window CAGR summary for stability/consistency signal.
- Per-scheme annualization factor (some debt/liquid funds publish 365/yr,
  equity funds publish ~252/yr); detected from actual NAV cadence.
- Filters zero/negative NAVs at SQL layer (data has some bogus rows).
"""

import logging
import math
import statistics
from datetime import date, datetime, timedelta

from mutual_fund_data import db_cursor

logger = logging.getLogger(__name__)

# India 10y G-Sec yield ~6.5% — used as risk-free for Sharpe.
RISK_FREE_RATE = 0.065

# Validity floors — return None for any metric where the underlying data
# is too thin to be meaningful.
MIN_NAVS_FOR_ANY_METRIC = 30
MIN_NAVS_FOR_VOLATILITY = 60
MIN_WINDOW_COVERAGE     = 0.80   # require 80% of requested window present


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_nav_series(scheme_code, max_years=5):
    """Return [(date, float)] ascending. Filters zero/negative NAVs."""
    cutoff = date.today() - timedelta(days=int(max_years * 365.25) + 30)
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT nav_date, nav_value::float AS nav
            FROM nav_history
            WHERE scheme_code = %s
              AND nav_date >= %s
              AND nav_value > 0
            ORDER BY nav_date ASC
            """,
            (scheme_code, cutoff),
        )
        return [(r['nav_date'], r['nav']) for r in cur.fetchall()]


def _years_between(d1, d2):
    return (d2 - d1).days / 365.25


def _cagr(start_nav, end_nav, years):
    """CAGR as decimal (0.12 = 12%). None if inputs invalid."""
    if start_nav is None or end_nav is None:
        return None
    if start_nav <= 0 or end_nav <= 0 or years <= 0:
        return None
    return (end_nav / start_nav) ** (1.0 / years) - 1.0


def _point_to_point_cagr(navs, years_back):
    """CAGR over last `years_back` years. None if window too sparse."""
    if not navs:
        return None
    end_date, end_nav = navs[-1]
    target_date = end_date - timedelta(days=int(years_back * 365.25))
    start_idx = next((i for i, (d, _) in enumerate(navs) if d >= target_date), None)
    if start_idx is None or start_idx >= len(navs) - 1:
        return None
    start_date, start_nav = navs[start_idx]
    actual_years = _years_between(start_date, end_date)
    if actual_years < years_back * MIN_WINDOW_COVERAGE:
        return None
    return _cagr(start_nav, end_nav, actual_years)


def _navs_per_year(navs):
    """Observed NAV cadence (used for annualization)."""
    span_years = _years_between(navs[0][0], navs[-1][0])
    if span_years <= 0:
        return None
    return len(navs) / span_years


def _rolling_cagr(navs, window_years):
    """Rolling CAGR over sliding windows. Returns list of decimal CAGRs."""
    if len(navs) < MIN_NAVS_FOR_ANY_METRIC:
        return []
    npy = _navs_per_year(navs)
    if npy is None or _years_between(navs[0][0], navs[-1][0]) < window_years:
        return []
    window = max(20, int(window_years * npy))
    if window >= len(navs):
        return []
    # Sample ~30 evenly-spaced windows rather than every NAV (faster, equivalent signal)
    step = max(1, (len(navs) - window) // 30)
    cagrs = []
    for i in range(0, len(navs) - window, step):
        s_date, s_nav = navs[i]
        e_date, e_nav = navs[i + window]
        c = _cagr(s_nav, e_nav, _years_between(s_date, e_date))
        if c is not None:
            cagrs.append(c)
    return cagrs


def _annualized_volatility(navs):
    """Annualized stdev of log returns. Decimal (0.15 = 15%)."""
    if len(navs) < MIN_NAVS_FOR_VOLATILITY:
        return None
    log_returns = []
    for i in range(1, len(navs)):
        prev_v = navs[i - 1][1]
        curr_v = navs[i][1]
        if prev_v > 0 and curr_v > 0:
            log_returns.append(math.log(curr_v / prev_v))
    if len(log_returns) < MIN_NAVS_FOR_VOLATILITY:
        return None
    npy = _navs_per_year(navs)
    if npy is None:
        return None
    return statistics.stdev(log_returns) * math.sqrt(npy)


def _max_drawdown(navs):
    """Peak-to-trough max decline with recovery info."""
    if len(navs) < 2:
        return None
    peak_v = navs[0][1]
    peak_d = navs[0][0]
    worst_dd = 0.0
    worst_peak_d = peak_d
    worst_peak_v = peak_v
    worst_trough_d = peak_d
    for d, v in navs:
        if v > peak_v:
            peak_v = v
            peak_d = d
        if peak_v > 0:
            dd = (v - peak_v) / peak_v
            if dd < worst_dd:
                worst_dd = dd
                worst_peak_d = peak_d
                worst_peak_v = peak_v
                worst_trough_d = d
    # Recovery: first NAV at or above worst_peak_v after trough
    days_to_recover = None
    for d, v in navs:
        if d <= worst_trough_d:
            continue
        if v >= worst_peak_v:
            days_to_recover = (d - worst_trough_d).days
            break
    return {
        'max_dd_pct':       round(worst_dd * 100, 2),
        'peak_date':        worst_peak_d.isoformat(),
        'peak_nav':         round(worst_peak_v, 4),
        'trough_date':      worst_trough_d.isoformat(),
        'drawdown_days':    (worst_trough_d - worst_peak_d).days,
        'days_to_recover':  days_to_recover,  # None = not yet recovered
    }


def _sharpe(cagr_decimal, vol_decimal, rf=RISK_FREE_RATE):
    if cagr_decimal is None or vol_decimal is None or vol_decimal <= 0:
        return None
    return round((cagr_decimal - rf) / vol_decimal, 2)


def _rolling_stats(cagrs):
    if not cagrs:
        return None
    return {
        'mean_pct':     round(statistics.mean(cagrs) * 100, 2),
        'min_pct':      round(min(cagrs) * 100, 2),
        'max_pct':      round(max(cagrs) * 100, 2),
        'stdev_pct':    round(statistics.stdev(cagrs) * 100, 2) if len(cagrs) >= 2 else 0.0,
        'window_count': len(cagrs),
    }


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_fund(scheme_code):
    """
    Full analytics for a single scheme. Raises ValueError if scheme not found.
    Returns dict with status='ok' or status='insufficient_data'.
    """
    scheme_code = str(scheme_code).strip()
    if not scheme_code:
        raise ValueError("scheme_code is required")

    with db_cursor() as cur:
        cur.execute(
            """
            SELECT scheme_code, scheme_name, amc_name, category, sub_category
            FROM schemes
            WHERE scheme_code = %s
            """,
            (scheme_code,),
        )
        scheme = cur.fetchone()
        if not scheme:
            raise ValueError(f"Scheme {scheme_code} not found")

    navs = _fetch_nav_series(scheme_code, max_years=5)

    if len(navs) < MIN_NAVS_FOR_ANY_METRIC:
        return {
            'status':       'insufficient_data',
            'scheme_code':  scheme['scheme_code'],
            'scheme_name':  scheme['scheme_name'],
            'amc_name':     scheme['amc_name'],
            'category':     scheme['category'],
            'nav_count':    len(navs),
            'message':      f"Only {len(navs)} NAV records (need ≥{MIN_NAVS_FOR_ANY_METRIC}).",
        }

    cagr_1y = _point_to_point_cagr(navs, 1)
    cagr_3y = _point_to_point_cagr(navs, 3)
    cagr_5y = _point_to_point_cagr(navs, 5)

    rolling_1y = _rolling_cagr(navs, 1)
    rolling_3y = _rolling_cagr(navs, 3)

    vol = _annualized_volatility(navs)
    dd  = _max_drawdown(navs)

    # Sharpe uses the best available headline CAGR
    sharpe_cagr = cagr_5y if cagr_5y is not None else (cagr_3y if cagr_3y is not None else cagr_1y)
    sharpe = _sharpe(sharpe_cagr, vol)

    return {
        'status':       'ok',
        'scheme_code':  scheme['scheme_code'],
        'scheme_name':  scheme['scheme_name'],
        'amc_name':     scheme['amc_name'],
        'category':     scheme['category'],
        'sub_category': scheme['sub_category'],
        'latest_nav': {
            'date':  navs[-1][0].isoformat(),
            'value': round(navs[-1][1], 4),
        },
        'data_coverage': {
            'oldest_nav_date':  navs[0][0].isoformat(),
            'newest_nav_date':  navs[-1][0].isoformat(),
            'nav_count':        len(navs),
            'years_of_history': round(_years_between(navs[0][0], navs[-1][0]), 2),
            'navs_per_year':    round(_navs_per_year(navs), 1),
        },
        'returns': {
            'cagr_1y_pct':     round(cagr_1y * 100, 2) if cagr_1y is not None else None,
            'cagr_3y_pct':     round(cagr_3y * 100, 2) if cagr_3y is not None else None,
            'cagr_5y_pct':     round(cagr_5y * 100, 2) if cagr_5y is not None else None,
            'rolling_1y_cagr': _rolling_stats(rolling_1y),
            'rolling_3y_cagr': _rolling_stats(rolling_3y),
        },
        'risk': {
            'annual_volatility_pct':    round(vol * 100, 2) if vol is not None else None,
            'max_drawdown':             dd,
            'sharpe_ratio':             sharpe,
            'risk_free_rate_used_pct':  round(RISK_FREE_RATE * 100, 2),
        },
        'computed_at': datetime.now().isoformat(timespec='seconds'),
    }
