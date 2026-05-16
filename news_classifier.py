"""
News Classifier — Gemini-powered headline classification (Module A)
====================================================================

Designed to be a drop-in replacement for keyword sentiment scoring.
Same output schema as news_sentiment.get_sentiment_score() so the rest
of the pipeline doesn't need to change.

Architecture:
  classify_headlines(headlines, mode='sentiment') → dict
  - mode='sentiment' : current production use, returns -3..+3 per headline
  - mode='structural': reserved for Module B (thesis confirmation), TBD

Caching:
  - In-memory dict, persisted to /tmp/headline_cache.json (Render-friendly)
  - Cached by SHA256 hash of headline text
  - 7-day TTL — headline classifications don't change with time

Failover:
  - If GEMINI_API_KEY missing or API errors, falls back to keyword scoring
    via news_sentiment.score_headline(). Pipeline never breaks.
"""

import os
import json
import hashlib
import time
import threading
import logging

logger = logging.getLogger('graham.news_classifier')

# ── Configuration ────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL    = 'gemini-flash-latest'  # always-current free tier model
CACHE_PATH      = os.environ.get('HEADLINE_CACHE_PATH',
                                  '/tmp/headline_cache.json' if os.path.exists('/tmp')
                                  else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'headline_cache.json'))
CACHE_TTL       = 7 * 24 * 3600  # 7 days

_cache = None
_cache_lock = threading.Lock()
_cache_dirty = False


def _load_cache():
    """Load cache from disk on first use."""
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        try:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, 'r') as f:
                    _cache = json.load(f)
                # Drop expired entries on load
                now = time.time()
                _cache = {k: v for k, v in _cache.items()
                          if (now - v.get('ts', 0)) < CACHE_TTL}
                logger.info(f"news_classifier cache loaded: {len(_cache)} entries")
            else:
                _cache = {}
        except Exception as e:
            logger.warning(f"news_classifier cache load failed: {e}")
            _cache = {}
    return _cache


def _save_cache():
    """Persist cache to disk. Called periodically, not every write."""
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        with _cache_lock:
            with open(CACHE_PATH, 'w') as f:
                json.dump(_cache, f)
            _cache_dirty = False
    except Exception as e:
        logger.warning(f"news_classifier cache save failed: {e}")


def _hash_headline(text: str) -> str:
    """Stable hash for cache key. Prefix is bumped on prompt schema changes
    to invalidate stale classifications."""
    return 'v4_' + hashlib.sha256(text.strip().lower().encode('utf-8')).hexdigest()[:16]


def _fallback_score(headline: str) -> dict:
    """Keyword-based fallback when Gemini is unavailable."""
    h_lower = headline.lower()

    # Filter out pure price-action headlines that the keyword scorer would mis-rank.
    # These describe price moves, not business changes.
    _price_keywords = [
        'falls', 'fall ', 'plunge', 'plunges', 'crashes', '52-week low', '52 week low',
        'block deal', 'block trade', 'bulk deal', 'bulk trade',
        'oversold', 'support', 'resistance', 'breakdown', 'breaks below',
        'hit a low', 'tanks', 'drops', 'slips', 'declines', 'decline ', 'declined',
        'down ', 'lower', 'dip', 'dips ', '% dip', 'after dip',
        'rallies', 'jumps', 'soars', 'rises', 'gain', 'up ', 'higher', 'surges',
        '52-week high', '52 week high', 'all-time high', 'all-time low',
        'profit-booking', 'profit booking', 'oversold bounce',
        'recovers', 'rebounds', 'bounce', 'bounced',
    ]
    _macro_keywords = [
        'nifty', 'sensex', 'fii outflow', 'fii inflow', 'market falls', 'market rises',
        'bank nifty', 'sectoral', 'index falls', 'index rises',
    ]
    _noise_keywords = [
        'should you buy', 'how to trade', 'trade spotlight', 'stocks to watch',
        'buzzing shares', 'top picks', 'expert view', 'target price',
    ]

    if any(k in h_lower for k in _noise_keywords):
        return {'score': 0, 'category': 'noise',
                'reason': 'keyword fallback: detected as noise'}
    if any(k in h_lower for k in _macro_keywords):
        return {'score': 0, 'category': 'macro_spillover',
                'reason': 'keyword fallback: detected as macro spillover'}
    if any(k in h_lower for k in _price_keywords):
        return {'score': 0, 'category': 'price_action',
                'reason': 'keyword fallback: detected as price action only'}

    # If none of the above filters match, default to neutral.
    # The keyword scorer is too aggressive on bank/finance headlines:
    # "screams buy after 25% dip" → it sees 'dip' and scores negative when actual sentiment is +2.
    # When in doubt, return 0. Better to under-react than to misread.
    return {
        'score':    0,
        'category': 'unclassified',
        'reason':   'keyword fallback (Gemini unavailable) — defaulting to neutral',
    }


def _build_prompt(headlines_with_symbols: list) -> str:
    """Build a single batched prompt for multiple headlines."""
    items = '\n'.join(
        f"{i+1}. [{sym}] {h}"
        for i, (sym, h) in enumerate(headlines_with_symbols)
    )
    return f"""Classify each Indian stock market headline below into a CATEGORY and a SENTIMENT SCORE.

CATEGORIES (pick one):
- structural: long-term thesis news (multi-year contracts, capacity expansion, sector tailwind validation, regulatory wins). Affects the company's intrinsic value.
- earnings: quarterly/annual results, guidance changes, profit/revenue updates, NPA disclosures, asset quality. Short-term significance.
- regulatory: SEBI notices, compliance issues, tax disputes, government policy changes. Can be material.
- corporate: management changes, M&A, dividends, buybacks, splits, IPOs, block deals where the deal itself is news.
- price_action: headlines describing price moves WITHOUT explaining a business cause — "stock falls X%", "hits 52W low", "shares plunge", "block deal of X shares", "stock under pressure", "support broken at Y", "bearish technical setup", "RSI oversold". These are short-term market mechanics, NOT business news. ALWAYS score 0 — price moves are an outcome, not a cause.
- noise: generic trading columns ("how to trade X", "buzzing stocks", "should you buy X"), advice columns, retail tip-sheets, target-price-only updates from analysts. ALWAYS score 0.
- macro_spillover: macro/sector news where this specific company is incidental ("Nifty falls, X among losers", "RBI policy weighs on banks", "FII outflows hit financials"). Company is mentioned only as an example of broader move. ALWAYS score 0 — these are macro, not company-specific.

CRITICAL DISTINCTION:
- "HDFC Bank Q2 NPAs rise to 2.1%" → earnings, score -2 (real business deterioration)
- "HDFC Bank falls 3% on profit-booking" → price_action, score 0 (just a price move)
- "HDFC Bank wins ₹500cr GST contract" → structural, score +3 (real business news)
- "HDFC Bank stock hits 52-week low" → price_action, score 0 (price not cause)
- "HDFC Bank RBI fines for KYC violations" → regulatory, score -2 (real consequence)

SENTIMENT (integer -3 to +3):
- +3: very positive (major contract win, blockbuster earnings, regulatory approval)
- +2: positive (good results, positive guidance)
- +1: mildly positive (small positive, analyst upgrade)
-  0: neutral or noise/price_action/macro_spillover (ALWAYS 0 for these categories)
- -1: mildly negative (analyst downgrade, minor concern)
- -2: negative (missed earnings, regulatory probe started)
- -3: very negative (fraud allegation, major contract loss, going-concern doubt)

OUTPUT: Return ONLY a valid JSON array, one object per headline in input order, no commentary, no markdown.
RESPONSE FORMAT (JSON array, one object per headline):
[
  {{
    "headline": "...",
    "category": "structural|earnings|regulatory|corporate|price_action|noise|macro_spillover",
    "score": int(-3..+3),
    "reason": "brief",
    "customer_entity": "<company/entity name if headline names a contract counterparty, client, or revenue partner, else null>"
  }},
  ...
]

CUSTOMER ENTITY EXTRACTION RULES:
- Extract a customer/counterparty name ONLY when the headline explicitly names them in connection with a contract, order, deal, or revenue relationship with the subject company.
- Examples that DO extract:
  - "Techno Electric wins ₹460 Cr IndiGrid contract" → customer_entity: "IndiGrid"
  - "Mazagon Dock signs deal with Indian Navy for 7 destroyers" → customer_entity: "Indian Navy"
  - "TCS bags multi-year contract with Singapore Airlines" → customer_entity: "Singapore Airlines"
  - "DIXON to supply mobile components to Xiaomi India" → customer_entity: "Xiaomi"
- Examples that do NOT extract (return null):
  - "Techno Electric Q2 profit up 24%" → null (no customer named)
  - "Stock falls 3% on profit-booking" → null (price action)
  - "Adani Group's Carmichael coal mine clearance pending" → null (regulatory event, no buying counterparty)
  - "Tata Motors among top picks" → null (analyst opinion, no transaction)
- Use the customer's most recognizable short name (e.g. "Indian Navy" not "Government of India Defence Ministry Navy")
- For multiple customers in one headline, pick the most prominent (only one entity per headline)
- If unsure, return null. False extractions hurt more than missed ones.

HEADLINES:
{items}

JSON:"""


def _call_gemini(headlines_with_symbols: list) -> list:
    """Send a batch to Gemini and parse the response. Returns list of dicts.
    Retries once on timeout (Render's outbound is slower than local)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    prompt = _build_prompt(headlines_with_symbols)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        }
    }

    last_err = None
    for attempt in (1, 2):
        try:
            r = requests.post(url, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            text = data['candidates'][0]['content']['parts'][0]['text']
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Gemini returned non-list: {type(parsed)}")
            if len(parsed) != len(headlines_with_symbols):
                raise ValueError(f"Gemini returned {len(parsed)} items, expected {len(headlines_with_symbols)}")
            return parsed
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt == 1:
                logger.warning(f"Gemini timeout on attempt 1, retrying once...")
                continue
            raise
        except Exception:
            raise  # non-network errors fail immediately

    if last_err:
        raise last_err


def classify_headlines(symbol: str, headlines: list, mode: str = 'sentiment') -> dict:
    """
    Classify a list of headlines for a single stock.
    Returns dict matching news_sentiment.get_sentiment_score() schema PLUS
    a 'classifications' list with per-headline detail.

    mode='sentiment'  : current production behavior
    mode='structural' : reserved for Module B (not yet implemented)
    """
    if not headlines:
        return {
            'symbol': symbol,
            'sentiment_score': 0,
            'sentiment_label': 'neutral',
            'positive_count': 0, 'negative_count': 0, 'neutral_count': 0,
            'total_articles': 0,
            'top_headlines': [],
            'classifications': [],
        }

    cache = _load_cache()
    global _cache_dirty
    now = time.time()

    # Determine what's cached vs needs classification
    to_classify = []      # list of (symbol, headline) to send to Gemini
    cached_results = {}   # idx → result dict
    for i, h in enumerate(headlines):
        h_hash = _hash_headline(h)
        entry = cache.get(h_hash)
        if entry and (now - entry.get('ts', 0)) < CACHE_TTL:
            cached_results[i] = entry['result']
        else:
            to_classify.append((i, symbol, h, h_hash))

    # Batch-call Gemini for uncached headlines
    if to_classify:
        try:
            batch = [(t[1], t[2]) for t in to_classify]
            gemini_results = _call_gemini(batch)
            for (orig_i, sym, h, h_hash), result in zip(to_classify, gemini_results):
                # Validate shape
                if not isinstance(result, dict):
                    result = _fallback_score(h)
                else:
                    score_v = result.get('score', 0)
                    try:
                        result['score'] = max(-3, min(3, int(score_v)))
                    except Exception:
                        result['score'] = 0
                    result.setdefault('category', 'noise')
                    result.setdefault('reason', '')
                cached_results[orig_i] = result
                with _cache_lock:
                    cache[h_hash] = {'result': result, 'ts': now}
                    _cache_dirty = True
        except Exception as e:
            # Log structured error so we can spot quota issues, model deprecation, etc.
            err_type = type(e).__name__
            err_msg  = str(e)[:200]
            logger.error(f"[Gemini ERROR] {symbol} | {err_type}: {err_msg}")

            # If timeout, try again with smaller batches (5 headlines instead of 10)
            # Smaller payloads = faster Gemini response = lower timeout risk
            if 'timeout' in err_type.lower() or 'timeout' in err_msg.lower():
                logger.info(f"[Gemini RECOVER] {symbol}: retrying with smaller batches")
                try:
                    half = len(to_classify) // 2 + 1
                    batch1 = [(t[1], t[2]) for t in to_classify[:half]]
                    batch2 = [(t[1], t[2]) for t in to_classify[half:]] if to_classify[half:] else []
                    results1 = _call_gemini(batch1) if batch1 else []
                    results2 = _call_gemini(batch2) if batch2 else []
                    for (orig_i, sym, h, h_hash), result in zip(to_classify, results1 + results2):
                        if not isinstance(result, dict):
                            result = _fallback_score(h)
                        else:
                            score_v = result.get('score', 0)
                            try:
                                result['score'] = max(-3, min(3, int(score_v)))
                            except Exception:
                                result['score'] = 0
                            result.setdefault('category', 'noise')
                            result.setdefault('reason', '')
                        cached_results[orig_i] = result
                        with _cache_lock:
                            cache[h_hash] = {'result': result, 'ts': now}
                            _cache_dirty = True
                    logger.info(f"[Gemini RECOVER] {symbol}: smaller-batch retry succeeded")
                    _save_cache()
                    # Skip the keyword fallback below since recovery worked
                    weighted_sum = 0.0; weight_total = 0.0
                    pos_n = neg_n = neu_n = 0
                    classifications = []
                    for i, h in enumerate(headlines):
                        r = cached_results.get(i, {'score': 0, 'category': 'noise', 'reason': ''})
                        cat = r.get('category', 'noise')
                        sc  = float(r.get('score') or 0)
                        if cat in ('noise', 'macro_spillover', 'price_action'):
                            weight = 0
                        elif cat == 'structural':
                            weight = 2.0
                        else:
                            weight = 1.0
                        if weight > 0:
                            weighted_sum += sc * weight
                            weight_total += weight
                        if   sc >  0: pos_n += 1
                        elif sc <  0: neg_n += 1
                        else:         neu_n += 1
                        classifications.append({
                            'headline': h, 'category': cat,
                            'score': sc, 'reason': r.get('reason', ''),
                        })
                    if weight_total > 0:
                        agg_score = round((weighted_sum / weight_total) * 10, 1)
                    else:
                        agg_score = 0.0
                    label = 'positive' if agg_score > 8 else 'negative' if agg_score < -8 else 'neutral'
                    top = sorted(classifications, key=lambda c: -abs(c['score']))[:3]
                    top_headlines = [{
                        'title':     c['headline'],
                        'sentiment': 'positive' if c['score'] > 0 else 'negative' if c['score'] < 0 else 'neutral',
                        'score':     c['score'],
                    } for c in top]
                    # Module B: log structural events from recovery path too
                    try:
                        from thesis_store import log_classifications_batch
                        log_classifications_batch(symbol, classifications)
                    except Exception as _e:
                        logger.warning(f"thesis_store log failed for {symbol}: {_e}")

                    return {
                        'symbol':           symbol,
                        'sentiment_score':  agg_score,
                        'sentiment_label':  label,
                        'positive_count':   pos_n,
                        'negative_count':   neg_n,
                        'neutral_count':    neu_n,
                        'total_articles':   len(headlines),
                        'top_headlines':    top_headlines,
                        'classifications':  classifications,
                        'fetched_at':       __import__('datetime').datetime.now().isoformat(),
                    }
                except Exception as e2:
                    logger.warning(f"[Gemini RECOVER FAIL] {symbol}: smaller-batch also failed: {e2}")
                    # fall through to keyword fallback below

            # Detect specific failure modes and log louder so they're greppable in Render logs
            if '429' in err_msg or 'quota' in err_msg.lower() or 'rate' in err_msg.lower():
                logger.error(f"[Gemini RATE LIMIT] {symbol} — falling back. Free tier reset at midnight Pacific.")
            elif '404' in err_msg or 'not found' in err_msg.lower():
                logger.error(f"[Gemini MODEL DEPRECATED] {symbol} — model {GEMINI_MODEL} returned 404. Update GEMINI_MODEL.")
            elif '401' in err_msg or '403' in err_msg or 'unauthorized' in err_msg.lower() or 'permission' in err_msg.lower():
                logger.error(f"[Gemini AUTH FAILED] {symbol} — check GEMINI_API_KEY env var on Render.")
            for (orig_i, sym, h, h_hash) in to_classify:
                cached_results[orig_i] = _fallback_score(h)

    # Save cache opportunistically (every call when dirty)
    _save_cache()

    # Aggregate using "Module A" rules:
    # - 'noise' and 'macro_spillover' contribute 0 (filtered out)
    # - 'structural' double-weighted (it matters more for fair value)
    # - others contribute their raw score
    weighted_sum = 0.0
    weight_total = 0.0
    pos_n = neg_n = neu_n = 0
    classifications = []
    for i, h in enumerate(headlines):
        r = cached_results.get(i, {'score': 0, 'category': 'noise', 'reason': ''})
        cat = r.get('category', 'noise')
        sc  = float(r.get('score') or 0)

        # Enforce zero-weight for categories that must never affect company sentiment
        if cat in ('noise', 'macro_spillover', 'price_action'):
            sc = 0.0  # clamp score too — don't trust Gemini to always return 0
            weight = 0  # short-term price moves and macro noise don't reflect business
        elif cat == 'structural':
            weight = 2.0  # thesis-relevant news weighted higher
        else:
            weight = 1.0

        if weight > 0:
            weighted_sum += sc * weight
            weight_total += weight

        if   sc >  0: pos_n += 1
        elif sc <  0: neg_n += 1
        else:         neu_n += 1

        classifications.append({
            'headline': h,
            'category': cat,
            'score':    sc,
            'reason':   r.get('reason', ''),
        })

    # Scale to roughly the same range as the keyword scorer (-30..+30)
    if weight_total > 0:
        avg = weighted_sum / weight_total
        agg_score = round(avg * 10, 1)  # -3..+3 → -30..+30
    else:
        agg_score = 0.0

    label = 'positive' if agg_score >  8 else 'negative' if agg_score < -8 else 'neutral'

    # Top 3 by absolute weighted score for display
    top = sorted(classifications, key=lambda c: -abs(c['score']))[:3]
    top_headlines = [{
        'title':     c['headline'],
        'sentiment': 'positive' if c['score'] > 0 else 'negative' if c['score'] < 0 else 'neutral',
        'score':     c['score'],
    } for c in top]

    # ── Module B: log structural events for future thesis-tracking ──
    # Fire-and-forget — never let logging failures break sentiment.
    try:
        from thesis_store import log_classifications_batch
        log_classifications_batch(symbol, classifications)
    except Exception as _e:
        logger.warning(f"thesis_store log failed for {symbol}: {_e}")

    return {
        'symbol':           symbol,
        'sentiment_score':  agg_score,
        'sentiment_label':  label,
        'positive_count':   pos_n,
        'negative_count':   neg_n,
        'neutral_count':    neu_n,
        'total_articles':   len(headlines),
        'top_headlines':    top_headlines,
        'classifications':  classifications,  # Module B will use this later
        'fetched_at':       __import__('datetime').datetime.now().isoformat(),
    }


def get_sentiment_score(symbol: str, headlines: list = None) -> dict:
    """
    Drop-in replacement for news_sentiment.get_sentiment_score().
    If headlines aren't supplied, fetches them via the original module.
    """
    if headlines is None:
        try:
            from news_sentiment import get_news
            raw_headlines = get_news(symbol)
            # get_news may return list of strings or list of dicts; normalize to strings
            headlines = []
            for item in raw_headlines or []:
                if isinstance(item, str):
                    headlines.append(item)
                elif isinstance(item, dict):
                    headlines.append(item.get('title') or item.get('headline') or str(item))
                else:
                    headlines.append(str(item))
            logger.info(f"news_classifier: fetched {len(headlines)} headlines for {symbol}")
        except Exception as e:
            logger.warning(f"news_classifier: get_news failed for {symbol}: {e}; falling back to original scorer")
            try:
                from news_sentiment import get_sentiment_score as orig
                return orig(symbol)
            except Exception:
                return {'symbol': symbol, 'sentiment_score': 0, 'sentiment_label': 'neutral'}

    if not headlines:
        return {
            'symbol': symbol, 'sentiment_score': 0, 'sentiment_label': 'neutral',
            'positive_count': 0, 'negative_count': 0, 'neutral_count': 0,
            'total_articles': 0, 'top_headlines': [], 'classifications': [],
        }

    return classify_headlines(symbol, headlines, mode='sentiment')
