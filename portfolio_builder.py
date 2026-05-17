"""
Portfolio Builder — Goal-based stock portfolio construction
============================================================

Inputs: amount, horizon, risk_appetite, goal, plus optional constraints.
Output: weighted stock portfolio with expected return range, sector breakdown,
and a backtest-style reference.

DESIGN PRINCIPLES:
- Hybrid scoring: nightly_cache for fast filtering, then /stock-analysis test_client
  for the final 15-20 candidates. Numbers users see in portfolio MUST match what
  they see on the individual stock page.
- Diversification: max 12-15% per stock, max 25% per sector, min 6-8 sectors.
- Risk-aware: conservative profiles include cash buffer, exclude high-vol stocks.
- Honest expectations: returns shown as ranges, not point estimates.

NON-GOALS (for now):
- Mean-variance optimization (needs historical covariance matrix)
- True Sharpe ratio targeting (needs return time series)
- Monte Carlo drawdown simulation
"""

import logging
import math
from collections import defaultdict

logger = logging.getLogger('graham.portfolio_builder')


# ── Profile configurations ───────────────────────────────────────────────────
# Each profile defines hard filters + soft preferences for stock selection.
PROFILES = {
    'Conservative': {
        'min_score':         62,        # only solid stocks
        'max_per_stock':     0.08,       # max 8% in any one stock
        'cash_buffer_pct':   0.15,       # 15% cash buffer
        'allow_pre_profit':  False,
        'allow_value_trap':  False,
        'prefer_dividend':   True,        # bonus for dividend payers
        'prefer_low_vol':    True,        # prefer Low Risk stocks
        'min_market_cap':    'large',    # only large caps
        'target_stocks':     15,         # 15 stocks for stability
        'sector_max':        0.22,
    },
    'Balanced': {
        'min_score':         58,
        'max_per_stock':     0.10,
        'cash_buffer_pct':   0.08,
        'allow_pre_profit':  False,
        'allow_value_trap':  False,
        'prefer_dividend':   True,
        'prefer_low_vol':    False,
        'min_market_cap':    'mid',
        'target_stocks':     12,
        'sector_max':        0.25,
    },
    'Growth': {
        'min_score':         55,
        'max_per_stock':     0.12,
        'cash_buffer_pct':   0.03,
        'allow_pre_profit':  False,
        'allow_value_trap':  False,
        'prefer_dividend':   False,
        'prefer_low_vol':    False,
        'min_market_cap':    'mid',
        'target_stocks':     12,
        'sector_max':        0.28,
    },
    'Aggressive': {
        'min_score':         52,
        'max_per_stock':     0.15,
        'cash_buffer_pct':   0.00,
        'allow_pre_profit':  True,
        'allow_value_trap':  False,
        'prefer_dividend':   False,
        'prefer_low_vol':    False,
        'min_market_cap':    'any',
        'target_stocks':     10,
        'sector_max':        0.30,
    },
}

# Goal modifiers — adjust the profile based on what user wants the portfolio to achieve
GOAL_MODIFIERS = {
    'Wealth preservation': {
        'min_score_bonus':    3,          # raise the floor
        'dividend_bonus':     0.20,       # heavy dividend bias
        'volatility_penalty': 0.30,       # punish high-vol stocks
        'tailwind_bonus':     0.05,
    },
    'Steady income': {
        'min_score_bonus':    0,
        'dividend_bonus':     0.30,       # heaviest dividend bias
        'volatility_penalty': 0.20,
        'tailwind_bonus':     0.00,
    },
    'Capital growth': {
        'min_score_bonus':    0,
        'dividend_bonus':     0.00,
        'volatility_penalty': 0.10,
        'tailwind_bonus':     0.20,       # favor structural growth stories
    },
    'Maximum returns': {
        'min_score_bonus':   -3,          # accept more risk
        'dividend_bonus':    -0.10,       # mild penalty (low-div = high-growth)
        'volatility_penalty': 0.00,       # don't penalize vol
        'tailwind_bonus':     0.30,       # heavy structural growth bias
    },
}

# Horizon adjustments — long horizon emphasizes fundamentals, short emphasizes ML
HORIZON_WEIGHTS = {
    1:  {'ml': 0.40, 'fundamentals': 0.30, 'valuation': 0.30},  # 1Y: ML matters most
    3:  {'ml': 0.30, 'fundamentals': 0.40, 'valuation': 0.30},
    5:  {'ml': 0.20, 'fundamentals': 0.50, 'valuation': 0.30},  # 5Y: fundamentals dominate
    10: {'ml': 0.10, 'fundamentals': 0.60, 'valuation': 0.30},  # 10Y: pure fundamentals
}


# ── Market cap helpers ───────────────────────────────────────────────────────
def _parse_market_cap_cr(mc_str):
    """Parse market cap string like '₹14.99T Cr' or '₹1.5L Cr' → numeric Crores."""
    if not mc_str or not isinstance(mc_str, str):
        return 0
    s = mc_str.replace('₹', '').replace(' Cr', '').replace(',', '').strip()
    try:
        if s.endswith('T'):
            return float(s[:-1]) * 1000  # T (thousand crore) — 1T = 1000 Cr
        elif s.endswith('L'):
            return float(s[:-1]) * 100000  # L (lakh crore) — 1L Cr = 100000 Cr
        elif s.endswith('K'):
            return float(s[:-1]) * 1000
        return float(s)
    except (ValueError, TypeError):
        return 0


def _market_cap_bucket(mc_cr):
    """Classify market cap. Indian convention:
       Large: >50,000 Cr | Mid: 5,000-50,000 Cr | Small: <5,000 Cr"""
    if mc_cr >= 50000:   return 'large'
    elif mc_cr >= 5000:  return 'mid'
    elif mc_cr > 0:      return 'small'
    return 'unknown'


def _meets_market_cap_filter(mc_cr, required):
    if required == 'any':    return True
    if required == 'large':  return mc_cr >= 50000
    if required == 'mid':    return mc_cr >= 5000
    if required == 'small':  return mc_cr > 0
    return True


# ── Risk and volatility helpers ──────────────────────────────────────────────
def _estimate_volatility(stock_data):
    """Approximate volatility from available signals.
       Returns 0-1 score (0 = low vol, 1 = high vol)."""
    risk_str = (stock_data.get('combined', {}).get('risk', 'Medium') or 'Medium').lower()
    risk_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very high': 1.0}
    base = risk_map.get(risk_str, 0.5)

    # RSI extremes indicate near-term volatility
    rsi = stock_data.get('ml', {}).get('rsi') or 50
    if rsi > 75 or rsi < 25:
        base = min(1.0, base + 0.15)

    return base


# ── Core selection algorithm ─────────────────────────────────────────────────
def build_portfolio(amount, horizon, risk_appetite, goal,
                    include_sectors=None, exclude_sectors=None,
                    exclude_psu=False, exclude_high_debt=False,
                    exclude_loss_makers=True, exclude_high_pledge=False,
                    min_market_cap='any', min_score=None,
                    force_include=None, force_exclude=None,
                    nightly_cache=None, app=None, csv_data=None):
    """
    Build a portfolio matching the user's criteria.

    Args:
        amount (float):       Investment amount in INR
        horizon (int):        Years (1, 3, 5, 10)
        risk_appetite (str):  'Conservative' | 'Balanced' | 'Growth' | 'Aggressive'
        goal (str):           'Wealth preservation' | 'Steady income' | 'Capital growth' | 'Maximum returns'
        include_sectors:      list of sectors to include (None = all)
        exclude_sectors:      list of sectors to exclude
        exclude_psu:          bool
        exclude_high_debt:    bool (excludes D/E > 1 for non-banks)
        exclude_loss_makers:  bool
        exclude_high_pledge:  bool (excludes >30% promoter pledge)
        min_market_cap:       'large' | 'mid' | 'small' | 'any'
        min_score:            int or None (None = use profile default)
        force_include:        list of symbols to force-include
        force_exclude:        list of symbols to never include
        nightly_cache:        dict from get_nightly_cache()
        app:                  Flask app for test_client calls
        csv_data:             pandas DataFrame of screener_fundamentals.csv

    Returns:
        dict with portfolio details and metadata
    """
    if risk_appetite not in PROFILES:
        return {'error': f"Invalid risk_appetite: {risk_appetite}. Use one of {list(PROFILES.keys())}"}
    if goal not in GOAL_MODIFIERS:
        return {'error': f"Invalid goal: {goal}. Use one of {list(GOAL_MODIFIERS.keys())}"}
    if horizon not in HORIZON_WEIGHTS:
        return {'error': f"Invalid horizon: {horizon}. Use one of {list(HORIZON_WEIGHTS.keys())}"}
    if amount < 10000:
        return {'error': 'Minimum investment amount: ₹10,000'}

    profile = dict(PROFILES[risk_appetite])
    modifier = GOAL_MODIFIERS[goal]

    if min_score is not None:
        profile['min_score'] = min_score
    else:
        profile['min_score'] += modifier['min_score_bonus']

    if min_market_cap != 'any':
        profile['min_market_cap'] = min_market_cap

    force_include  = [s.upper().strip() for s in (force_include or []) if s]
    force_exclude  = [s.upper().strip() for s in (force_exclude or []) if s]
    include_sectors = [s.lower().strip() for s in (include_sectors or [])]
    exclude_sectors = [s.lower().strip() for s in (exclude_sectors or [])]

    # ── Phase 1: Fast filter using nightly_cache ───────────────────────────────
    if not nightly_cache or not nightly_cache.get('stocks'):
        return {'error': 'Nightly cache unavailable — try again in a moment'}

    # Build approximate mcap lookup from CSV (networth_cr × pb_ratio proxy)
    _csv_mcap = {}
    if csv_data is not None:
        try:
            import pandas as _pd
            _df = csv_data
            for _, row in _df.iterrows():
                sym = str(row.get('symbol') or row.iloc[0] or '').strip().upper()
                if not sym:
                    continue
                try:
                    nw = float(row.get('networth_cr') or 0)
                    pb = float(row.get('pb') or row.get('price_to_book') or 0)
                    if nw > 0 and pb > 0:
                        _csv_mcap[sym] = nw * pb
                except Exception:
                    pass
        except Exception:
            pass

    # Mcap filter thresholds (generous — just reject obvious size mismatches)
    _mcap_min = {'large': 30000, 'mid': 3000, 'small': 0, 'any': 0}.get(
        profile.get('min_market_cap', 'any'), 0
    )

    candidates = []
    rejected = defaultdict(int)

    for sym, cached in nightly_cache['stocks'].items():
        sym_u = sym.upper()
        if sym_u in force_exclude:
            rejected['force_excluded'] += 1
            continue

        # Cheap mcap pre-filter using CSV approximation
        if sym_u not in force_include and _mcap_min > 0:
            approx_mcap = _csv_mcap.get(sym_u, 0)
            if approx_mcap > 0 and approx_mcap < _mcap_min:
                rejected['mcap_too_small'] += 1
                continue

        ml = cached.get('ml') or {}
        ml_score = ml.get('ml_score') or 50

        # Build a richer pre_rank using cache signals
        pre_rank = float(ml_score)

        # PEG adjustment: low PEG (value+growth) → boost; high PEG → trim
        val = cached.get('valuation') or {}
        pe  = val.get('pe') or val.get('current_pe') or 0
        eg  = val.get('earnings_growth') or val.get('eg') or 0
        if pe and eg and pe > 0 and eg > 0:
            peg = pe / eg
            if peg < 1.0:
                pre_rank += 5
            elif peg < 1.5:
                pre_rank += 2
            elif peg > 3.0:
                pre_rank -= 3

        # Revenue quality from chart_insights (if cached)
        chart = cached.get('chart_insights') or {}
        rev_q = (chart.get('revenue_quality') or '').lower()
        if 'strong' in rev_q or 'accelerating' in rev_q:
            pre_rank += 3
        elif 'declining' in rev_q or 'weak' in rev_q:
            pre_rank -= 3

        candidates.append({
            'symbol':       sym_u,
            'ml_score_fast': ml_score,
            'pre_rank':     pre_rank,
            'forced':       sym_u in force_include,
        })

    # Always include forced stocks even if filtered out above
    forced_present = set(c['symbol'] for c in candidates)
    for fs in force_include:
        if fs not in forced_present:
            candidates.append({'symbol': fs, 'ml_score_fast': 50, 'pre_rank': 50, 'forced': True})

    target_count = profile['target_stocks']
    pool_size = min(25, target_count * 2 + 5)

    candidates.sort(key=lambda c: (not c['forced'], -c['pre_rank']))
    top_for_analysis = candidates[:pool_size]

    # ── Phase 2: Full /stock-analysis via test_client for finalists ────────────
    analyzed = []
    if app is None:
        return {'error': 'Flask app reference required for full analysis'}

    import threading
    analyzed_lock = threading.Lock()

    def _analyze_one(sym):
        try:
            with app.test_client() as client:
                resp = client.get(f'/stock-analysis?symbol={sym}')
                if resp.status_code != 200:
                    return
                data = resp.get_json() or {}
            with analyzed_lock:
                analyzed.append((sym, data))
        except Exception as e:
            logger.warning(f"portfolio_builder analyze fail {sym}: {e}")

    threads = [threading.Thread(target=_analyze_one, args=(c['symbol'],))
               for c in top_for_analysis]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)

    # ── Phase 3: Apply hard filters on full data ───────────────────────────────
    survivors = []
    for sym, data in analyzed:
        combined  = data.get('combined') or {}
        quote     = data.get('quote') or {}
        fund      = data.get('fundamentals') or {}
        val_sig   = combined.get('valuation_signal') or {}

        industry = (quote.get('industry') or '').lower()

        if include_sectors and not any(s in industry for s in include_sectors):
            rejected['sector_not_in_include'] += 1
            continue
        if exclude_sectors and any(s in industry for s in exclude_sectors):
            rejected['sector_in_exclude'] += 1
            continue

        # PSU filter (heuristic: high promoter % + bank/utility/defence)
        if exclude_psu:
            prom = fund.get('promoter_pct') or 0
            psu_sectors = ['bank', 'power', 'defence', 'aerospace', 'oil', 'gas', 'mining', 'railway']
            if prom > 50 and any(s in industry for s in psu_sectors):
                rejected['psu_excluded'] += 1
                continue

        # Loss-maker filter
        if exclude_loss_makers:
            eps = fund.get('eps_latest')
            if eps is not None and eps < 0:
                rejected['loss_maker'] += 1
                continue
            if val_sig.get('label') == 'Loss-Making':
                rejected['loss_maker'] += 1
                continue

        # Pre-profit filter
        if not profile['allow_pre_profit']:
            if val_sig.get('label', '').startswith('Pre-Profit'):
                rejected['pre_profit'] += 1
                continue

        # Value trap filter
        if not profile['allow_value_trap']:
            if val_sig.get('label') == 'Value Trap Risk':
                rejected['value_trap'] += 1
                continue

        # Debt filter
        if exclude_high_debt:
            de = fund.get('screener_de')
            industry_is_bank = any(b in industry for b in ['bank', 'financial', 'insurance', 'finance'])
            if de is not None and de > 1.0 and not industry_is_bank:
                rejected['high_debt'] += 1
                continue

        # Pledge filter
        if exclude_high_pledge:
            pledge = fund.get('pledged_pct')
            if pledge is not None and pledge > 30:
                rejected['high_pledge'] += 1
                continue

        # Market cap filter
        mc_cr = _parse_market_cap_cr(quote.get('market_cap'))
        if not _meets_market_cap_filter(mc_cr, profile['min_market_cap']):
            rejected['below_market_cap'] += 1
            continue

        # Score filter (apply unless forced)
        is_forced = sym in force_include
        combined_score = combined.get('score') or 0
        if combined_score < profile['min_score'] and not is_forced:
            rejected['below_min_score'] += 1
            continue

        # Skip stocks with skip_reason (insurance/financial valuation N/A)
        if val_sig.get('skip_reason') and not is_forced:
            rejected['valuation_skip'] += 1
            continue

        survivors.append({
            'symbol':         sym,
            'data':           data,
            'combined_score': combined_score,
            'industry':       quote.get('industry') or 'Other',
            'market_cap_cr':  mc_cr,
            'mc_bucket':      _market_cap_bucket(mc_cr),
            'forced':         is_forced,
        })

    if len(survivors) < 5:
        return {
            'error': (f'Only {len(survivors)} stocks survived your filters. '
                      f'Loosen constraints (lower min_score, fewer excluded sectors, '
                      f'or pick a different risk profile).'),
            'rejected_summary': dict(rejected),
            'survivors_count': len(survivors),
        }

    # ── Phase 4: Goal-based composite scoring ─────────────────────────────────
    horizon_w = HORIZON_WEIGHTS[horizon]
    for s in survivors:
        data     = s['data']
        combined = data.get('combined') or {}
        ml       = data.get('ml') or {}
        val_sig  = combined.get('valuation_signal') or {}

        ml_s   = ml.get('ml_score') or 50
        fund_s = combined.get('screener_score') or 50
        pct_fair = val_sig.get('pct_vs_fair')
        val_s  = max(0, min(100, 50 - pct_fair * 1.0)) if pct_fair is not None else 50

        composite = (ml_s   * horizon_w['ml'] +
                     fund_s * horizon_w['fundamentals'] +
                     val_s  * horizon_w['valuation'])

        div_yield = (data.get('valuation') or {}).get('dividend_yield') or 0
        if isinstance(div_yield, str):
            try: div_yield = float(div_yield)
            except: div_yield = 0
        if div_yield > 0.02:
            composite *= (1 + modifier['dividend_bonus'])

        vol = _estimate_volatility(data)
        composite *= (1 - vol * modifier['volatility_penalty'])

        if val_sig.get('tailwind_multiplier', 1.0) > 1.05:
            composite *= (1 + modifier['tailwind_bonus'])

        s['composite_score'] = composite
        s['volatility']      = vol
        s['div_yield']       = div_yield
        s['has_tailwind']    = val_sig.get('tailwind_multiplier', 1.0) > 1.05

    # ── Phase 5: Sector-aware selection ────────────────────────────────────────
    survivors.sort(key=lambda s: (not s['forced'], -s['composite_score']))

    selected        = []
    selected_symbols = set()
    sector_counts   = defaultdict(int)
    target          = profile['target_stocks']
    invested_pct    = 1.0 - profile['cash_buffer_pct']
    max_per_sector_count = max(2, math.ceil(target * profile['sector_max']))

    # First pass: forced stocks
    for s in survivors:
        if s['forced'] and s['symbol'] not in selected_symbols:
            selected.append(s)
            selected_symbols.add(s['symbol'])
            sector_counts[s['industry']] += 1
            if len(selected) >= target:
                break

    # Second pass: fill with sector diversity
    for s in survivors:
        if len(selected) >= target:
            break
        if s['symbol'] in selected_symbols:
            continue
        if sector_counts[s['industry']] >= max_per_sector_count:
            continue
        selected.append(s)
        selected_symbols.add(s['symbol'])
        sector_counts[s['industry']] += 1

    # Third pass: relax sector cap BUT still respect min_score and valuation
    # Better to ship 10 high-quality stocks than 12 with a low-quality filler.
    if len(selected) < target:
        for s in survivors:
            if len(selected) >= target:
                break
            if s['symbol'] in selected_symbols:
                continue
            # Phase 3 already enforced min_score; if a stock got through, it's eligible.
            # But ALSO avoid stocks with overvalued labels in this fallback pass.
            val_label = (s['data'].get('combined') or {}).get('valuation_signal', {}).get('label', '')
            if val_label in ('Overvalued Quality', 'Overpriced Weak Business',
                             'Severely Overvalued Quality', 'Value Trap Risk'):
                continue
            selected.append(s)
            selected_symbols.add(s['symbol'])
            sector_counts[s['industry']] += 1

    # If we still don't have target, that's fine — quality over quantity.
    # Minimum 5 is enforced earlier; 10+ is the goal but not mandatory.

    if len(selected) < 5:
        return {
            'error': (f'Could not assemble at least 5 stocks with diversification. '
                      f'Selected only {len(selected)}.'),
            'rejected_summary': dict(rejected),
        }

    # ── Phase 6: Weighting ─────────────────────────────────────────────────────
    total_score = sum(s['composite_score'] for s in selected)
    for s in selected:
        s['raw_weight'] = (s['composite_score'] / total_score
                           if total_score > 0 else 1.0 / len(selected))

    # Cap individual weights
    max_w = profile['max_per_stock']
    capped = True
    iterations = 0
    while capped and iterations < 20:
        capped = False
        iterations += 1
        over = [s for s in selected if s['raw_weight'] > max_w]
        if not over:
            break
        excess = sum(s['raw_weight'] - max_w for s in over)
        for s in over:
            s['raw_weight'] = max_w
        under = [s for s in selected if s['raw_weight'] < max_w]
        if under:
            under_total = sum(s['raw_weight'] for s in under) or 1
            for s in under:
                s['raw_weight'] += excess * (s['raw_weight'] / under_total)
            capped = True

    # Cap sector weights
    sector_w = defaultdict(float)
    for s in selected:
        sector_w[s['industry']] += s['raw_weight']

    sector_cap = profile['sector_max']
    for sector, w in list(sector_w.items()):
        if w > sector_cap:
            scale = sector_cap / w
            for s in selected:
                if s['industry'] == sector:
                    s['raw_weight'] *= scale

    # Normalize to invested_pct
    total_w = sum(s['raw_weight'] for s in selected)
    if total_w > 0:
        for s in selected:
            s['raw_weight'] = s['raw_weight'] * invested_pct / total_w

    # ── Phase 7: Allocations ───────────────────────────────────────────────────
    holdings = []
    for s in selected:
        d       = s['data']
        quote   = d.get('quote') or {}
        combined = d.get('combined') or {}
        val_sig = combined.get('valuation_signal') or {}
        forecast = d.get('forecast') or {}
        price   = quote.get('price') or 0
        weight  = s['raw_weight']
        rupees  = round(amount * weight, 0)
        shares  = math.floor(rupees / price) if price > 0 else 0
        actual_invested = shares * price

        holdings.append({
            'symbol':          s['symbol'],
            'company_name':    quote.get('company_name') or s['symbol'],
            'industry':        s['industry'],
            'mc_bucket':       s['mc_bucket'],
            'price':           price,
            'weight_pct':      round(weight * 100, 2),
            'allocated_inr':   rupees,
            'shares':          shares,
            'actual_invested': round(actual_invested, 0),
            'combined_score':  round(s['composite_score'], 1),
            'grade':           combined.get('grade'),
            'verdict':         combined.get('verdict'),
            'val_signal':      val_sig.get('label'),
            'risk':            combined.get('risk'),
            'fair_value':      val_sig.get('fair_value'),
            'pct_vs_fair':     val_sig.get('pct_vs_fair'),
            'target_1y':       (forecast.get('1y') or {}).get('price_target'),
            'tailwind_theme':  val_sig.get('tailwind_theme'),
            'reason_selected': _build_reason(s, profile, modifier),
        })

    # ── Phase 8: Portfolio-level metrics ──────────────────────────────────────
    cash_inr        = round(amount * profile['cash_buffer_pct'], 0)
    total_invested  = sum(h['actual_invested'] for h in holdings)
    leftover        = amount - total_invested - cash_inr

    sector_breakdown = defaultdict(lambda: {'count': 0, 'weight': 0.0, 'inr': 0})
    for h in holdings:
        sector_breakdown[h['industry']]['count']  += 1
        sector_breakdown[h['industry']]['weight'] += h['weight_pct']
        sector_breakdown[h['industry']]['inr']    += h['actual_invested']

    expected_return = _estimate_expected_return(holdings, horizon, risk_appetite)

    avg_vol    = sum(s['volatility'] for s in selected) / len(selected)
    risk_label = ('Low'    if avg_vol < 0.35 else
                  'Medium' if avg_vol < 0.60 else 'High')

    return {
        'status':   'ok',
        'profile':  risk_appetite,
        'goal':     goal,
        'horizon_y': horizon,
        'amount':   amount,
        'holdings': holdings,
        'portfolio_summary': {
            'total_stocks':     len(holdings),
            'total_invested':   total_invested,
            'cash_buffer':      cash_inr,
            'leftover_unspent': round(leftover, 0),
            'sector_count':     len(sector_breakdown),
            'avg_score':        round(sum(h['combined_score'] for h in holdings) / len(holdings), 1),
            'portfolio_risk':   risk_label,
        },
        'sector_breakdown': dict(sector_breakdown),
        'expected_return':  expected_return,
        'rejected_summary': dict(rejected),
        'disclaimers': [
            'Educational tool only — not personalized investment advice.',
            'Past performance does not guarantee future returns.',
            'Returns shown are model estimates; actual returns may vary significantly.',
            'Consult a SEBI-registered advisor before making investment decisions.',
            'This portfolio assumes you can hold for the full horizon without rebalancing.',
        ],
    }


def _build_reason(survivor, profile, modifier):
    """Generate a short human-readable reason this stock was selected."""
    reasons = []
    data    = survivor['data']
    combined = data.get('combined') or {}
    val_sig = combined.get('valuation_signal') or {}

    score = combined.get('score') or 0
    if score >= 75:   reasons.append('strong overall score')
    elif score >= 65: reasons.append('solid overall score')

    if val_sig.get('tailwind_multiplier', 1.0) > 1.05:
        reasons.append(f"structural tailwind ({val_sig.get('tailwind_theme', 'sector growth')})")

    if survivor.get('div_yield', 0) > 0.02:
        reasons.append(f"{round(survivor['div_yield']*100, 1)}% dividend")

    if val_sig.get('pct_vs_fair') and val_sig['pct_vs_fair'] < -15:
        reasons.append(f"{abs(round(val_sig['pct_vs_fair']))}% below fair value")

    if survivor.get('forced'):
        reasons.append('user requested')

    return ', '.join(reasons[:3]) if reasons else 'meets profile criteria'


def _estimate_expected_return(holdings, horizon, risk_appetite):
    """Honest expected return range based on price targets and risk profile."""
    if not holdings:
        return {'low': 0, 'mid': 0, 'high': 0, 'note': 'no holdings'}

    weighted_implied_return = 0.0
    total_weight = 0.0
    for h in holdings:
        price  = h.get('price') or 0
        target = h.get('target_1y') or 0
        weight = h.get('weight_pct', 0) / 100
        if price > 0 and target > 0:
            implied_1y = (target - price) / price
            weighted_implied_return += implied_1y * weight
            total_weight += weight

    if total_weight > 0:
        weighted_implied_return /= total_weight
    else:
        weighted_implied_return = 0.10  # default 10% assumption

    annual_mid = max(0.04, min(0.22, weighted_implied_return))

    vol_band = {'Conservative': 0.04, 'Balanced': 0.06, 'Growth': 0.09, 'Aggressive': 0.12}
    band = vol_band.get(risk_appetite, 0.07)

    low_annual  = max(0.0, annual_mid - band)
    high_annual = annual_mid + band

    return {
        'low_annual_pct':  round(low_annual  * 100, 1),
        'mid_annual_pct':  round(annual_mid  * 100, 1),
        'high_annual_pct': round(high_annual * 100, 1),
        'cumulative_low':  round(((1 + low_annual)  ** horizon - 1) * 100, 1),
        'cumulative_mid':  round(((1 + annual_mid)  ** horizon - 1) * 100, 1),
        'cumulative_high': round(((1 + high_annual) ** horizon - 1) * 100, 1),
        'note': (f'Range based on individual stock 1Y price targets + '
                 f'{risk_appetite.lower()} profile volatility band'),
    }
