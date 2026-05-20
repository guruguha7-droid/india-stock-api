"""
News Sentiment Engine
======================
Fetches Google News RSS headlines for NSE stocks
and scores them positive/negative/neutral.
Free — no API key needed.

Scoring model:
- Lifecycle-tiered patterns for capex/orders/regulatory (announcement < completion < early)
- Per-keyword negation proximity check ("no revenue growth" → penalty, not bonus)
- Per-keyword forward-looking discount ("expects profit" → +1, not +3)
- Negated negative keywords flip to positive ("no layoffs" → +1)
- Deal-size magnitude bonus on order matches
- Weighted-average aggregation (stronger signals dominate noise)
- Near-duplicate deduplication (same story from 5 outlets counts once)
"""

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import time
from datetime import datetime
import re

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ── Proximity helpers ─────────────────────────────────────────────────────────

_NEG_PAT = re.compile(
    r'\b(no|not|without|never|neither|nor|fails?\s+to|unable\s+to|denies?|rules?\s+out)\b'
)
_FWD_PAT = re.compile(
    r'\b(plans?\s+to|expects?\s+to|may|could|might|likely|targeting|aims?\s+to|'
    r'proposes?|considering|mulling|eyeing|to\s+invest|intends?\s+to)\b'
)

def _is_negated(h: str, kw_start: int, window: int = 35) -> bool:
    """True if a negation word appears within `window` chars before the keyword."""
    return bool(_NEG_PAT.search(h[max(0, kw_start - window): kw_start]))

def _is_forward(h: str, kw_start: int, window: int = 45) -> bool:
    """True if a forward-looking qualifier appears within `window` chars before the keyword."""
    return bool(_FWD_PAT.search(h[max(0, kw_start - window): kw_start]))

def _lifecycle_score(pts: int, h: str, match_start: int) -> int:
    """
    Apply probabilistic modifiers to a lifecycle pattern match.

    Negation  → flips the signal:
      "no order cancellation" (-6) → +2   (good news)
      "no plant commissioned" (+6) → -1   (concerning)

    Forward-looking → discounts by ~half (uncertain outcome):
      "may commission plant"  (+6) → +3
      "may cancel contract"   (-6) → -3
      "plans to invest"       (+3) → +2   (already discounted tier, floor at +1)
    """
    if _is_negated(h, match_start):
        # Flip: negative event negated → mild positive; positive event negated → mild negative
        return +2 if pts < 0 else -1
    if _is_forward(h, match_start):
        # Discount: halve the score, keep sign, floor at ±1
        halved = pts // 2
        return max(1, halved) if pts > 0 else min(-1, halved)
    return pts


# ── Deal / order size magnitude bonus ────────────────────────────────────────

_CRORE_RE   = re.compile(r'(?:₹|rs\.?)\s*([\d,]+(?:\.\d+)?)\s*(?:cr(?:ore)?|k\s*cr)', re.I)
_LAKH_CR_RE = re.compile(r'([\d,]+(?:\.\d+)?)\s*lakh\s*crore', re.I)

def _deal_size_bonus(h: str) -> int:
    m = _LAKH_CR_RE.search(h)
    if m:
        return 4                       # multi-lakh crore = landmark
    m = _CRORE_RE.search(h)
    if m:
        try:
            amt = float(m.group(1).replace(',', ''))
            if amt >= 10_000: return 3
            if amt >= 5_000:  return 2
            if amt >= 1_000:  return 1
        except ValueError:
            pass
    return 0


# ── Lifecycle-tiered patterns ─────────────────────────────────────────────────
# Each group is checked first-match-wins to avoid double-counting.
# Announcement (uncertain) < On-time completion (full credit) < Early (bonus)

_CAPEX_TIERS = [
    # Early / ahead of schedule — best signal
    (re.compile(
        r'ahead of schedule|early completion|commissioned ahead|delivered early|'
        r'before deadline|beats deadline', re.I), +7),
    # Commissioned / operational — completion confirmed
    (re.compile(
        r'commissions?\s+plant|plant commissioned|capacity commissioned|'
        r'inaugurates?\s+plant|plant inaugurated|operationalises?|goes?\s+live|'
        r'begins?\s+production|starts?\s+commercial production|'
        r'commences?\s+operations?|plant\s+operati|unit\s+commissioned', re.I), +6),
    # Cancellation / scrapped — worst outcome
    (re.compile(
        r'cancels?\s+(?:plant|project|expansion|capex|investment)|'
        r'scraps?\s+(?:plant|project|expansion|capex)|'
        r'abandons?\s+(?:plant|project|expansion)|'
        r'capex\s+scrapped|writes?\s+off\s+capex|project\s+cancelled|'
        r'shelves?\s+(?:capex|expansion|plant|project)', re.I), -7),
    # Delay / overrun — negative
    (re.compile(
        r'capex delayed|project delayed|cost overrun|behind schedule|'
        r'construction halted|project stalled|time overrun|capex cut|'
        r'reduces?\s+capex|cuts?\s+capex\s+guidance', re.I), -5),
    # Announcement — discount for uncertainty
    (re.compile(
        r'plans?\s+capex|greenfield|brownfield|capacity expansion plan|'
        r'announces?\s+investment|new plant plan|sets?\s+up\s+(?:plant|facility)|'
        r'capex plan|capex target|capital expenditure plan|capex guidance|'
        r'to\s+invest\b|capacity addition', re.I), +3),
]

_ORDER_TIERS = [
    # Cancellation / termination — checked first; deal size applies as extra penalty
    (re.compile(
        r'order\s+cancell|contract\s+(?:cancell|terminat|rescind)|'
        r'cancels?\s+(?:order|contract)|terminates?\s+(?:order|contract)|'
        r'loses?\s+(?:order|contract|bid)|contract\s+lost|order\s+lost|'
        r'loses?\s+deal|deal\s+(?:falls?\s+through|called\s+off|collapse|cancell|terminat)|'
        r'fails?\s+to\s+(?:win|secure|bag|renew)\s+(?:order|contract)|'
        r'contract\s+goes?\s+to\s+rival|order\s+goes?\s+to\s+competitor|'
        r'jv\s+dissolved|partnership\s+(?:cancell|terminat|ends?)', re.I), -6),
    # Delivery confirmed
    (re.compile(
        r'delivers?\s+order|order\s+(?:executed|fulfilled|completed|shipped)|'
        r'supplies?\s+to\b|dispatches?\s+order', re.I), +6),
    # Win / bag / secure
    (re.compile(
        r'wins?\s+(?:mega|large|massive|record|landmark|₹|rs\.?)?\s*order|'
        r'bags?\s+(?:order|contract)|secures?\s+(?:order|contract)|'
        r'receives?\s+(?:large|major|significant|repeat)?\s*(?:order|contract)|'
        r'gets?\s+(?:order|contract\s+from)|order\s+from\b', re.I), +4),
    # Pipeline / book — lagging signal
    (re.compile(r'order\s+(?:pipeline|book|inflow|backlog)', re.I), +2),
]

# ── M&A / deal lifecycle ──────────────────────────────────────────────────────
# Separate group because M&A events don't overlap cleanly with order/capex patterns.

_DEAL_TIERS = [
    # Merger / acquisition falls through
    (re.compile(
        r'merger\s+(?:called\s+off|cancell|terminat|collapse|abandon|fails?)|'
        r'acquisition\s+(?:called\s+off|cancell|scrapped|abandon|collapse)|'
        r'deal\s+(?:falls?\s+through|called\s+off|collapse|cancell|scrapped)|'
        r'takeover\s+(?:called\s+off|cancell|collapse|abandon)', re.I), -6),
    # Merger / acquisition announced — moderate positive (uncertain outcome)
    (re.compile(
        r'acquires?\b|agrees?\s+to\s+acquire|merger\s+(?:deal|agree|announc)|'
        r'takeover\s+(?:bid|offer|deal)|strategic\s+acquisition', re.I), +3),
    # Merger / acquisition completed
    (re.compile(
        r'acquisition\s+(?:complete|closed?|finalis)|merger\s+(?:complete|closed?|finalis)|'
        r'successfully\s+acquires?|completes?\s+acquisition', re.I), +5),
]

_REGULATORY_TIERS = [
    # Hard ban / debarment
    (re.compile(r'sebi\s+ban|sebi\s+bars?|debarred|exchanges?\s+bans?', re.I), -8),
    # Notice / probe
    (re.compile(
        r'sebi\s+(?:notice|probe|investigation|scrutiny|show\s+cause)|'
        r'ed\s+(?:notice|summons|raid|arrest)|income\s+tax\s+(?:notice|raid|search)|'
        r'cbi\s+(?:raid|probe|arrest)', re.I), -6),
    # Approval — confirmed
    (re.compile(
        r'fda\s+(?:approval|approved|clearance)|drug\s+approval|'
        r'regulatory\s+approval|gets?\s+(?:cdsco|dcgi)\s+approval|'
        r'nod\s+from\s+(?:fda|cdsco|dcgi|sebi|rbi)', re.I), +6),
    # FDA warning / rejection
    (re.compile(
        r'fda\s+(?:warning|import\s+alert|483|inspection\s+fail)|'
        r'cdsco\s+(?:notice|reject)|drug\s+recall', re.I), -6),
]


# ── Strong / catastrophic one-shot signals ────────────────────────────────────

STRONG_POSITIVE = [
    'record profit', 'all time high', 'all-time high', 'blockbuster',
    'landmark deal', 'massive order', 'fda breakthrough', 'index inclusion',
    'nifty 50 inclusion', 'sensex inclusion', 'highest ever profit',
    'highest ever revenue', 'debt free', 'zero debt',
]

STRONG_NEGATIVE = [
    'fraud', 'scam', 'money laundering', 'ponzi', 'forensic audit',
    'whistleblower', 'promoter arrested', 'md arrested', 'ceo arrested',
    'bankruptcy', 'insolvency', 'nclt', 'default on',
]


# ── Regular keyword dictionaries (no overlap with lifecycle patterns) ─────────

POSITIVE_KEYWORDS = [
    # Financial results
    'revenue growth', 'beat estimates', 'strong results', 'record revenue',
    'profit growth', 'margin expansion', 'ebitda growth',
    # Analyst actions
    'upgrade', 'buy rating', 'target raised', 'price target raised',
    'raises target', 'overweight', 'outperform', 'strong buy',
    'accumulate', 'positive outlook', 'bullish', 'multibagger',
    # Corporate actions
    'dividend', 'buyback', 'share buyback', 'bonus shares', 'rights issue',
    # Market signals
    'fii buying', 'dii buying', 'promoter buying', 'institutional buying',
    '52-week high', 'fresh high', 'breakout', 'adrs jump', 'adrs surge',
    # Business
    'licensing deal', 'new product launch',
]

NEGATIVE_KEYWORDS = [
    # Financial results
    'profit warning', 'revenue decline', 'weak results', 'miss estimates',
    'below expectations', 'disappoints', 'margin pressure', 'ebitda decline',
    # Analyst actions
    'downgrade', 'sell rating', 'target cut', 'price target cut',
    'lowers target', 'bearish', 'avoid', 'caution',
    # Corporate red flags
    'promoter selling', 'promoter pledge', 'pledge increase',
    'rising debt', 'debt default', 'npa', 'write-off',
    'layoffs', 'job cuts',
    # Market signals
    '52-week low', 'adrs drop', 'adrs fall', 'sell off',
    # Macro / legal
    'tariff', 'sanction', 'regulatory risk', 'recall',
    'lawsuit', 'litigation', 'court order', 'arbitration loss',
    # Investor losses
    'sitting on a loss', 'erodes wealth',
]


# ── Headline scorer ───────────────────────────────────────────────────────────

def score_headline(headline: str) -> int:
    """
    Score a single headline. Returns integer in [-10, +10].

    Priority order:
    1. Lifecycle patterns (capex / order / regulatory) — first match per group
    2. Strong signals (one-shot ±8)
    3. Context patterns (crash, nosedive, etc.)
    4. Regular keywords with per-keyword negation / forward-looking check
    """
    h = headline.lower()
    score = 0

    # ── 1. Lifecycle patterns (with negation + forward-looking modifiers) ────
    for pattern, pts in _CAPEX_TIERS:
        m = pattern.search(h)
        if m:
            score += _lifecycle_score(pts, h, m.start())
            break

    for pattern, pts in _ORDER_TIERS:
        m = pattern.search(h)
        if m:
            adjusted = _lifecycle_score(pts, h, m.start())
            score += adjusted
            # deal size amplifies wins and penalises cancellations
            bonus = _deal_size_bonus(h)
            score += bonus if adjusted > 0 else -bonus
            break

    for pattern, pts in _DEAL_TIERS:
        m = pattern.search(h)
        if m:
            score += _lifecycle_score(pts, h, m.start())
            break

    for pattern, pts in _REGULATORY_TIERS:
        m = pattern.search(h)
        if m:
            score += _lifecycle_score(pts, h, m.start())
            break

    # ── 2. Strong signals ─────────────────────────────────────────────────
    for kw in STRONG_POSITIVE:
        if kw in h:
            score += 8

    for kw in STRONG_NEGATIVE:
        if kw in h:
            score -= 8

    # ── 3. Context patterns ───────────────────────────────────────────────
    if re.search(r'fall.{0,20}gains|drop.{0,20}gains|decline.{0,20}gains', h):
        score -= 4

    if 'sitting on a loss' in h or 'investors on a loss' in h:
        score -= 8

    if any(w in h for w in ['nosedive', 'carnage', 'bloodbath']):
        score -= 6

    if re.search(r'\btanks?\b', h):
        score -= 6

    if 'crash' in h and any(w in h for w in ['stock', 'share', 'nifty', 'sensex']):
        score -= 6

    if re.search(r'\bplunges?\b', h):
        score -= 5

    # ── 4. Regular keywords with negation / forward-looking per match ─────
    for kw in POSITIVE_KEYWORDS:
        pos = h.find(kw)
        if pos >= 0:
            if _is_negated(h, pos):
                score -= 2      # "no revenue growth" → penalty
            elif _is_forward(h, pos):
                score += 1      # "expects strong results" → mild positive
            else:
                score += 3

    for kw in NEGATIVE_KEYWORDS:
        pos = h.find(kw)
        if pos >= 0:
            if _is_negated(h, pos):
                score += 1      # "no layoffs", "no debt default" → mild positive
            else:
                score -= 3

    return max(-10, min(10, score))


# ── Near-duplicate deduplication ─────────────────────────────────────────────

def _dedupe(articles: list) -> list:
    """Remove near-duplicate headlines (same story from multiple outlets)."""
    seen = set()
    out = []
    for a in articles:
        key = re.sub(r'\s+', ' ', a['title'].lower()[:55]).strip()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


# ── Fetcher ───────────────────────────────────────────────────────────────────

def get_news(symbol: str, company_name: str = None, max_articles: int = 12) -> list:
    """
    Fetch Google News RSS headlines for a stock.
    Returns list of dicts with title, score, url, date.
    """
    query = f"{company_name} NSE stock India" if company_name else f"{symbol} NSE stock India"
    url = (f"https://news.google.com/rss/search"
           f"?q={requests.utils.quote(query)}"
           f"&hl=en-IN&gl=IN&ceid=IN:en")

    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        soup  = BeautifulSoup(r.content, 'html.parser')
        items = soup.find_all('item')[:max_articles]

        articles = []
        for item in items:
            title = item.title.text if item.title else ''
            link  = item.link.next_sibling.strip() if item.link else ''
            pub   = item.pubdate.text if item.pubdate else ''
            s     = score_headline(title)
            articles.append({
                'title':     title,
                'score':     s,
                'url':       link,
                'date':      pub,
                'published': datetime.now().isoformat(),
                'sentiment': 'positive' if s > 0 else 'negative' if s < 0 else 'neutral',
            })

        return _dedupe(articles)

    except Exception:
        return []


# ── Aggregator ────────────────────────────────────────────────────────────────

def get_sentiment_score(symbol: str, company_name: str = None) -> dict:
    """
    Get aggregated sentiment score for a stock.
    Returns dict with overall score and breakdown.

    Aggregation: weighted average where weight = abs(score) + 1, so a single
    strong signal (+8 or -8) dominates over several weak neutrals.
    """
    articles = get_news(symbol, company_name, max_articles=12)

    if not articles:
        return {
            'symbol':          symbol,
            'sentiment_score': 0,
            'sentiment_label': 'neutral',
            'positive_count':  0,
            'negative_count':  0,
            'neutral_count':   0,
            'total_articles':  0,
            'top_headlines':   [],
        }

    scores  = [a['score'] for a in articles]
    weights = [abs(s) + 1 for s in scores]   # neutrals weight=1, max weight=11
    wavg    = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    normalised = round(wavg * 10, 1)
    normalised = max(-100, min(100, normalised))

    pos = sum(1 for s in scores if s > 0)
    neg = sum(1 for s in scores if s < 0)
    neu = sum(1 for s in scores if s == 0)

    if normalised > 10:    label = 'positive'
    elif normalised < -10: label = 'negative'
    else:                  label = 'neutral'

    return {
        'symbol':          symbol,
        'sentiment_score': normalised,
        'sentiment_label': label,
        'positive_count':  pos,
        'negative_count':  neg,
        'neutral_count':   neu,
        'total_articles':  len(articles),
        'top_headlines':   [
            {'title': a['title'][:120], 'sentiment': a['sentiment'], 'score': a['score']}
            for a in sorted(articles, key=lambda x: abs(x['score']), reverse=True)[:3]
        ],
    }


def get_sentiment_all(symbols: list, delay: float = 1.0) -> dict:
    """Get sentiment for all stocks. Returns dict keyed by symbol."""
    results = {}
    print(f"\n Fetching news sentiment for {len(symbols)} stocks...")
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2d}/{len(symbols)}] {sym}...", end=' ', flush=True)
        result = get_sentiment_score(sym)
        results[sym] = result
        print(f"Score: {result['sentiment_score']:+.0f} | "
              f"{result['sentiment_label']} | {result['total_articles']} articles")
        if i < len(symbols):
            time.sleep(delay)
    return results


if __name__ == "__main__":
    test = ['MAZDOCK', 'TATASTEEL', 'RELIANCE', 'BAJFINANCE', 'SUNPHARMA']

    print("=" * 60)
    print("  News Sentiment Engine — Test Run")
    print("=" * 60)

    for sym in test:
        result = get_sentiment_score(sym)
        print(f"\n{sym} — Score: {result['sentiment_score']:+.0f} "
              f"({result['sentiment_label'].upper()})")
        print(f"  {result['positive_count']}+ | {result['negative_count']}- | "
              f"{result['neutral_count']}= | {result['total_articles']} total")
        for h in result['top_headlines']:
            icon = '+' if h['sentiment'] == 'positive' else '-' if h['sentiment'] == 'negative' else '='
            print(f"  [{icon}{h['score']:+d}] {h['title'][:90]}")
        time.sleep(1)
