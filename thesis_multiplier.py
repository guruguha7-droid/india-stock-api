"""
Module B Step 3 — Data-driven tailwind multipliers (with hardcoded fallback)
============================================================================

Replaces the static STRUCTURAL_TAILWINDS dict in api_server.py with a function
that reads from thesis_store and derives multipliers from accumulated history.

DESIGN PHILOSOPHY:
  When the store has insufficient data for a stock (< MIN_EVENTS_FOR_SIGNAL),
  we use the hardcoded dict as the source of truth — your editorial calls
  remain the safety net.

  When the store has enough data, we blend the hardcoded value with the
  data-derived value, weighting toward data as confidence grows.

  When the store has rich data (lots of structural events with recency),
  the function relies almost entirely on the data.

  This transitions smoothly from "hardcoded today" to "fully data-driven
  in 4-8 weeks" without any code change — just data accumulation.

PARAMETERS YOU CAN TUNE:
  MIN_EVENTS_FOR_SIGNAL — below this, fall back to hardcoded
  FULL_DATA_THRESHOLD   — above this, ignore hardcoded entirely
  CONFIRMATION_WINDOW_DAYS — how far back to look for "recent" confirmation
  DECAY_PER_QUIET_MONTH — multiplier penalty per quiet month
"""

import logging
import math

logger = logging.getLogger('graham.thesis_multiplier')

# ── Tunables ─────────────────────────────────────────────────────────────────
MIN_EVENTS_FOR_SIGNAL    = 5         # below this, use hardcoded only
FULL_DATA_THRESHOLD      = 30        # at/above this, use data only
CONFIRMATION_WINDOW_DAYS = 90        # "recent" = last 3 months
DECAY_PER_QUIET_MONTH    = 0.02      # -2% per quiet month after first
NEUTRAL_BASE             = 1.00      # default multiplier when no thesis info
MAX_MULTIPLIER           = 1.25      # cap on positive boost
MIN_MULTIPLIER           = 0.95      # floor on decay


def _compute_data_driven_multiplier(symbol, summary):
    """
    Derive a multiplier from accumulated thesis events.

    Logic:
      - Start at NEUTRAL_BASE (1.00)
      - Add for structural event density (more events = stronger thesis confirmation)
      - Add for recent confirmation (events in last 30 days count double)
      - Subtract for quiet periods (decay if no structural event in 30+ days)
      - Clamp to [MIN_MULTIPLIER, MAX_MULTIPLIER]
    """
    struct_count   = summary.get('structural_events', 0)
    days_since     = summary.get('days_since_last_structural')
    total_events   = summary.get('total_events', 0)

    if struct_count == 0:
        return NEUTRAL_BASE

    # Base lift from structural event density (caps quickly to avoid runaway)
    # 1 structural event → +0.5% boost from this term
    # 5 structural events → +2.4% (most stocks max out here)
    # 15+ structural events → +5.4% (heavy thesis confirmation)
    density_boost = 0.05 * math.log1p(struct_count)

    # Recency multiplier: events in last 30 days mean the thesis is "live"
    recency_factor = 1.0
    if days_since is not None:
        if days_since <= 30:
            recency_factor = 1.5   # actively confirming
        elif days_since <= 90:
            recency_factor = 1.0   # normal
        elif days_since <= 180:
            recency_factor = 0.5   # weakening
        else:
            recency_factor = 0.0   # thesis quiet

    confirmation_lift = density_boost * recency_factor

    # Decay penalty: if thesis has been quiet for months
    decay_penalty = 0.0
    if days_since is not None and days_since > 30:
        quiet_months = (days_since - 30) / 30.0
        decay_penalty = min(quiet_months * DECAY_PER_QUIET_MONTH, 0.10)

    mult = NEUTRAL_BASE + confirmation_lift - decay_penalty
    return max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, mult))


def get_thesis_multiplier(symbol, hardcoded_dict=None):
    """
    Returns (theme, multiplier) for a stock. Blends hardcoded editorial calls
    with data-driven signal based on data availability.

    Args:
      symbol: stock symbol like 'TECHNOE'
      hardcoded_dict: the existing STRUCTURAL_TAILWINDS dict (optional)
                      Format: {'TECHNOE': ('theme description', 1.18), ...}

    Returns:
      (theme_str | None, multiplier_float)

    Behavior:
      - No data + not in dict → (None, 1.00)
      - No data + in dict → (theme, hardcoded_mult)  [today's behavior for most stocks]
      - Some data + in dict → blend of both, weighted by data confidence
      - Lots of data + maybe in dict → purely data-driven
    """
    hardcoded = (hardcoded_dict or {}).get(symbol)
    hardcoded_theme = hardcoded[0] if hardcoded else None
    hardcoded_mult  = hardcoded[1] if hardcoded else NEUTRAL_BASE

    try:
        from thesis_store import get_summary
        summary = get_summary(symbol, days=180)
        total_events = summary.get('total_events', 0)
    except Exception as e:
        logger.warning(f"thesis_multiplier: store read failed for {symbol}: {e}")
        return (hardcoded_theme, hardcoded_mult)

    # No data → use hardcoded as-is
    if total_events < MIN_EVENTS_FOR_SIGNAL:
        return (hardcoded_theme, hardcoded_mult)

    data_mult = _compute_data_driven_multiplier(symbol, summary)

    # Confidence in the data scales linearly between MIN and FULL thresholds
    if total_events >= FULL_DATA_THRESHOLD:
        confidence = 1.0
    else:
        confidence = (total_events - MIN_EVENTS_FOR_SIGNAL) / (FULL_DATA_THRESHOLD - MIN_EVENTS_FOR_SIGNAL)

    # Blend: weighted average of hardcoded and data-driven
    blended_mult = (hardcoded_mult * (1 - confidence)) + (data_mult * confidence)

    # Annotate the theme with whether data is talking
    if confidence >= 0.5:
        theme = hardcoded_theme or 'derived from news flow'
        if hardcoded_theme:
            theme = f"{hardcoded_theme} (signal confirmed)"
    else:
        theme = hardcoded_theme

    return (theme, round(blended_mult, 3))


def get_thesis_health(symbol):
    """
    Returns a UI-friendly thesis health summary for badges.

    Returns:
      {
        'status': 'confirming' | 'quiet' | 'at_risk' | 'no_data',
        'label':  human-readable label,
        'color':  'green' | 'gold' | 'red' | 'gray',
        'last_event_days': int | None,
        'structural_count_90d': int,
        'detail': human-readable detail string,
      }
    """
    try:
        from thesis_store import get_summary
        summary = get_summary(symbol, days=CONFIRMATION_WINDOW_DAYS)
    except Exception as e:
        return {
            'status': 'no_data', 'label': 'no data',
            'color': 'gray', 'last_event_days': None,
            'structural_count_90d': 0,
            'detail': 'No thesis history yet — system still accumulating'
        }

    struct_count = summary.get('structural_events', 0)
    days_since   = summary.get('days_since_last_structural')
    total_events = summary.get('total_events', 0)

    if total_events < MIN_EVENTS_FOR_SIGNAL:
        return {
            'status': 'no_data', 'label': 'no data',
            'color': 'gray', 'last_event_days': days_since,
            'structural_count_90d': struct_count,
            'detail': f"Only {total_events} events tracked — need {MIN_EVENTS_FOR_SIGNAL}+ for signal"
        }

    if struct_count >= 3 and days_since is not None and days_since <= 60:
        return {
            'status': 'confirming', 'label': 'thesis confirming',
            'color': 'green', 'last_event_days': days_since,
            'structural_count_90d': struct_count,
            'detail': f"{struct_count} structural confirmations in last {CONFIRMATION_WINDOW_DAYS} days"
        }

    if days_since is not None and days_since > 180:
        return {
            'status': 'at_risk', 'label': 'thesis at risk',
            'color': 'red', 'last_event_days': days_since,
            'structural_count_90d': struct_count,
            'detail': f"No structural news in {int(days_since)} days — thesis may be weakening"
        }

    return {
        'status': 'quiet', 'label': 'thesis quiet',
        'color': 'gold', 'last_event_days': days_since,
        'structural_count_90d': struct_count,
        'detail': f"{struct_count} confirmations in last 90 days — neither strong nor weakening"
    }
