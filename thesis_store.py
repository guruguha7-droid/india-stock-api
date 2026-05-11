"""
Module B — Thesis Persistence Layer
====================================

Logs structural news events per stock with timestamps. Used by future
versions of the tailwind multiplier to make it data-driven instead of
hardcoded.

This module is intentionally write-only and read-helper at the moment.
It accumulates data; the multiplier-decay logic comes later.

Storage:
  - JSON file at /tmp/thesis_history.json (Render-friendly, tmp persists
    across requests but not deploys; for stable storage use Render Standard
    or override with THESIS_STORE_PATH env var)
  - Schema is intentionally flat — easy to inspect, debug, and migrate

Threading:
  - All writes go through _store_lock so concurrent requests don't corrupt
  - Saves are throttled (max once per 5s) to avoid hammering disk
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone

logger = logging.getLogger('graham.thesis_store')

# ── Configuration ────────────────────────────────────────────────────────────
STORE_PATH = os.environ.get(
    'THESIS_STORE_PATH',
    '/tmp/thesis_history.json' if os.path.exists('/tmp')
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_history.json')
)

# How long to keep events. Older events stop counting toward "thesis health"
# but stay in the file for audit/inspection.
EVENT_MAX_AGE_DAYS = 365

# Throttle disk writes to avoid burning IO on every event
SAVE_THROTTLE_SEC = 5.0

# Per-stock event cap to prevent unbounded growth (FIFO eviction)
MAX_EVENTS_PER_STOCK = 200

# Categories that count as "thesis confirmation" (positive structural signals)
CONFIRMING_CATEGORIES = {'structural'}

# ── State ────────────────────────────────────────────────────────────────────
_store = None             # in-memory dict, lazy-loaded
_store_lock = threading.RLock()
_dirty = False
_last_save = 0.0


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load():
    """Load the store from disk on first use."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        try:
            if os.path.exists(STORE_PATH):
                with open(STORE_PATH, 'r') as f:
                    _store = json.load(f)
                logger.info(f"thesis_store loaded: {len(_store)} stocks tracked")
            else:
                _store = {}
        except Exception as e:
            logger.warning(f"thesis_store load failed: {e}; starting empty")
            _store = {}
    return _store


def _save():
    """Persist the store to disk. Throttled — actual write happens at most once
    per SAVE_THROTTLE_SEC seconds."""
    global _dirty, _last_save
    if not _dirty:
        return
    now = time.time()
    if now - _last_save < SAVE_THROTTLE_SEC:
        return
    try:
        with _store_lock:
            tmp_path = STORE_PATH + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(_store, f, indent=None, separators=(',', ':'))
            os.replace(tmp_path, STORE_PATH)  # atomic on POSIX
            _dirty = False
            _last_save = now
    except Exception as e:
        logger.warning(f"thesis_store save failed: {e}")


def _prune_old(events):
    """Remove events older than EVENT_MAX_AGE_DAYS."""
    if not events:
        return events
    cutoff = time.time() - (EVENT_MAX_AGE_DAYS * 86400)
    fresh = []
    for ev in events:
        try:
            ev_time = datetime.fromisoformat(ev['ts'].replace('Z', '+00:00')).timestamp()
            if ev_time >= cutoff:
                fresh.append(ev)
        except Exception:
            fresh.append(ev)  # keep events with unparseable timestamps
    return fresh


def log_event(symbol, classification, headline, thesis_tag=None):
    """
    Record a single classified headline as an event for the given stock.

    Args:
      symbol (str): Stock symbol, e.g. 'TECHNOE'
      classification (dict): Output from news_classifier — must have 'category', 'score', 'reason'
      headline (str): The headline text
      thesis_tag (str, optional): Specific thesis being tracked (e.g. 'data_center_buildout').
                                   If None, tagged generically.
    """
    if not symbol or not isinstance(classification, dict):
        return
    cat = classification.get('category', 'noise')
    sc  = classification.get('score', 0)

    # Skip noise — don't log things that don't matter
    if cat in ('noise', 'macro_spillover', 'price_action'):
        return

    store = _load()
    with _store_lock:
        if symbol not in store:
            store[symbol] = {
                'events':         [],
                'last_updated':   _now_iso(),
                'lifetime_count': 0,
                'lifetime_structural_count': 0,
            }
        bucket = store[symbol]
        bucket['events'].append({
            'ts':       _now_iso(),
            'category': cat,
            'score':    sc,
            'headline': headline[:200],  # cap to avoid bloat
            'reason':   (classification.get('reason') or '')[:200],
            'thesis_tag': thesis_tag,
        })
        # Cap to MAX_EVENTS_PER_STOCK (FIFO)
        if len(bucket['events']) > MAX_EVENTS_PER_STOCK:
            bucket['events'] = bucket['events'][-MAX_EVENTS_PER_STOCK:]
        bucket['last_updated'] = _now_iso()
        bucket['lifetime_count'] += 1
        if cat in CONFIRMING_CATEGORIES:
            bucket['lifetime_structural_count'] += 1
        global _dirty
        _dirty = True

    _save()


def log_classifications_batch(symbol, classifications, thesis_tag=None):
    """Convenience: log a batch of classifications (e.g. all headlines for one stock)."""
    if not classifications:
        return
    for c in classifications:
        log_event(
            symbol=symbol,
            classification=c,
            headline=c.get('headline') or c.get('title') or '',
            thesis_tag=thesis_tag,
        )


def get_history(symbol, days=None):
    """
    Read all logged events for a stock.

    Args:
      symbol (str): e.g. 'TECHNOE'
      days (int, optional): Only return events within the last N days. None = all.

    Returns:
      List of event dicts, newest first.
    """
    store = _load()
    bucket = store.get(symbol, {})
    events = list(bucket.get('events') or [])
    if days is not None and events:
        cutoff = time.time() - (days * 86400)
        events = [
            ev for ev in events
            if _safe_ts(ev.get('ts', '')) >= cutoff
        ]
    events.sort(key=lambda e: e.get('ts', ''), reverse=True)
    return events


def get_summary(symbol, days=90):
    """
    Aggregate signal density for a stock over a window. Used by future
    multiplier logic to detect "thesis still being confirmed" vs "thesis quiet".

    Returns:
      {
        'symbol': str,
        'window_days': int,
        'total_events': int,
        'structural_events': int,
        'avg_score': float,            # mean of scores in window
        'last_structural_at': str|None,
        'days_since_last_structural': int|None,
        'recent_categories': dict,     # category → count
      }
    """
    events = get_history(symbol, days=days)
    structural = [e for e in events if e.get('category') in CONFIRMING_CATEGORIES]
    cat_counts = {}
    for e in events:
        c = e.get('category', 'unknown')
        cat_counts[c] = cat_counts.get(c, 0) + 1

    last_struct_at = structural[0]['ts'] if structural else None
    days_since = None
    if last_struct_at:
        try:
            last_ts = datetime.fromisoformat(last_struct_at.replace('Z', '+00:00')).timestamp()
            days_since = round((time.time() - last_ts) / 86400, 1)
        except Exception:
            pass

    avg_score = (sum(e.get('score', 0) for e in events) / len(events)) if events else 0.0

    return {
        'symbol':                       symbol,
        'window_days':                  days,
        'total_events':                 len(events),
        'structural_events':            len(structural),
        'avg_score':                    round(avg_score, 2),
        'last_structural_at':           last_struct_at,
        'days_since_last_structural':   days_since,
        'recent_categories':            cat_counts,
    }


def _safe_ts(s):
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except Exception:
        return 0


def health_check():
    """Diagnostic: how many stocks tracked, total events, store size."""
    store = _load()
    total_events = sum(len(b.get('events') or []) for b in store.values())
    try:
        size_bytes = os.path.getsize(STORE_PATH) if os.path.exists(STORE_PATH) else 0
    except Exception:
        size_bytes = 0
    return {
        'stocks_tracked':    len(store),
        'total_events':      total_events,
        'store_path':        STORE_PATH,
        'store_size_bytes':  size_bytes,
        'store_size_kb':     round(size_bytes / 1024, 1),
    }


def force_save():
    """For testing / shutdown — bypass throttle."""
    global _last_save
    _last_save = 0
    _save()
