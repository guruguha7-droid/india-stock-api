"""
News Sentiment Engine
======================
Fetches Google News RSS headlines for NSE stocks
and scores them positive/negative/neutral.
Free — no API key needed.
"""

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import time
from datetime import datetime
import re

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── Sentiment keyword dictionaries ────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    # Financial performance
    'profit', 'revenue growth', 'beats', 'beat estimates', 'record',
    'strong results', 'outperforms', 'upgrade', 'buy rating', 'target raised',
    'price target raised', 'raises target', 'strong growth', 'robust',
    'surge', 'jumps', 'rallies', 'gains', 'rises', 'climbs',
    # Business wins
    'new contract', 'wins order', 'large deal', 'acquisition', 'expansion',
    'partnership', 'launch', 'new product', 'fda approval', 'approval',
    'dividend', 'bonus', 'buyback', 'share buyback',
    # Analyst actions
    'overweight', 'outperform', 'strong buy', 'accumulate', 'positive outlook',
    'bullish', 'upside', 'multibagger', 'recommended',
    # Market actions
    'adrs jump', 'adrs surge', 'fresh high', '52-week high', 'breakout',
    'institutional buying', 'fii buying', 'promoter buying',
]

NEGATIVE_KEYWORDS = [
    # Financial performance
    'loss', 'decline', 'misses', 'miss estimates', 'below expectations',
    'profit warning', 'revenue decline', 'weak results', 'disappoints',
    'downgrade', 'sell rating', 'target cut', 'price target cut', 'lowers target',
    # Business problems
    'fraud', 'scam', 'investigation', 'sebi notice', 'penalty', 'fine',
    'layoffs', 'job cuts', 'restructuring', 'debt', 'default',
    'promoter selling', 'promoter pledge', 'pledge',
    # Market actions
    'plunge', 'crash', 'tumbles', 'falls', 'drops', 'slumps', 'tanks',
    'sell off', 'bearish', 'downside', 'caution', 'avoid',
    '52-week low', 'adrs drop', 'adrs fall',
    # Macro risks
    'tariff', 'sanction', 'regulatory risk', 'ban', 'recall',
    'lawsuit', 'litigation', 'court order',
    # Investor losses
    'investors sitting on a loss', 'sitting on a loss', 'erodes wealth',
]

STRONG_NEGATIVE = [
    'fraud', 'scam', 'sebi ban', 'default', 'bankruptcy', 'insolvency',
    'promoter arrested', 'cbi probe', 'ed probe', 'money laundering',
]

STRONG_POSITIVE = [
    'record profit', 'all time high', 'blockbuster', 'landmark deal',
    'massive order', 'fda breakthrough', 'index inclusion',
]


def score_headline(headline: str) -> int:
    """
    Score a single headline.
    Returns: -10 to +10
    """
    h = headline.lower()
    score = 0

    # ── Strong signals ────────────────────────────────────────────────
    for kw in STRONG_POSITIVE:
        if kw in h:
            score += 8

    for kw in STRONG_NEGATIVE:
        if kw in h:
            score -= 8

    # ── Context-aware checks ──────────────────────────────────────────
    # "fall after gains" = negative despite containing 'gains'
    if re.search(r'fall.{0,20}gains|drop.{0,20}gains|decline.{0,20}gains', h):
        score -= 4

    # "investors sitting on loss" = strong negative
    if 'sitting on a loss' in h or 'investors on a loss' in h:
        score -= 8

    # "nosedive" or "tanks" = strong negative
    if any(w in h for w in ['nosedive', 'tanks', 'tank', 'carnage', 'bloodbath']):
        score -= 6

    # "crash" with stocks = strong negative
    if 'crash' in h and any(w in h for w in ['stock', 'share', 'nifty', 'sensex']):
        score -= 6

    # "despite" negates positive — "up despite" = weak positive, "down despite" = negative
    if 'plunge' in h or 'plunges' in h:
        score -= 5

    # ── Regular signals ───────────────────────────────────────────────
    for kw in POSITIVE_KEYWORDS:
        if kw in h:
            score += 3

    for kw in NEGATIVE_KEYWORDS:
        if kw in h:
            score -= 3

    return max(-10, min(10, score))


def get_news(symbol: str, company_name: str = None, max_articles: int = 10) -> list:
    """
    Fetch Google News RSS headlines for a stock.
    Returns list of dicts with title, score, url, date.
    """
    # Build search query
    query = f"{symbol} NSE stock India"
    if company_name:
        query = f"{company_name} NSE stock India"

    url = (f"https://news.google.com/rss/search"
           f"?q={requests.utils.quote(query)}"
           f"&hl=en-IN&gl=IN&ceid=IN:en")

    try:
        r = requests.get(url, timeout=10,
                         headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        items = soup.find_all('item')[:max_articles]

        articles = []
        for item in items:
            title = item.title.text if item.title else ''
            link  = item.link.next_sibling.strip() if item.link else ''
            pub   = item.pubdate.text if item.pubdate else ''
            score = score_headline(title)
            articles.append({
                'title':     title,
                'score':     score,
                'url':       link,
                'date':      pub,
                'sentiment': 'positive' if score > 0 else 'negative' if score < 0 else 'neutral',
            })

        return articles

    except Exception:
        return []


def get_sentiment_score(symbol: str, company_name: str = None) -> dict:
    """
    Get aggregated sentiment score for a stock.
    Returns dict with overall score and breakdown.
    """
    articles = get_news(symbol, company_name, max_articles=10)

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

    scores    = [a['score'] for a in articles]
    avg       = sum(scores) / len(scores)

    # Normalise to -100 to +100
    normalised = round(avg * 10, 1)
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
        label = result['sentiment_label']
        score = result['sentiment_score']
        n     = result['total_articles']
        print(f"Score: {score:+.0f} | {label} | {n} articles")
        if i < len(symbols):
            time.sleep(delay)
    return results


if __name__ == "__main__":
    # Test on 5 stocks
    test = ['WIPRO', 'TCS', 'RELIANCE', 'BAJFINANCE', 'SUNPHARMA']

    print("="*60)
    print("  News Sentiment Engine — Test Run")
    print("="*60)

    for sym in test:
        result = get_sentiment_score(sym)
        print(f"\n{sym} — Score: {result['sentiment_score']:+.0f} "
              f"({result['sentiment_label'].upper()})")
        print(f"  {result['positive_count']} positive | "
              f"{result['negative_count']} negative | "
              f"{result['neutral_count']} neutral")
        print(f"  Top headlines:")
        for h in result['top_headlines']:
            icon = 'POS' if h['sentiment'] == 'positive' else 'NEG' if h['sentiment'] == 'negative' else 'NEU'
            print(f"    [{icon}] [{h['score']:+d}] {h['title'][:90]}")
        time.sleep(1)
