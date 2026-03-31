"""
Macro Sentiment Engine
=======================
Fetches macro/economic news that affects Indian stocks broadly
and applies sector-specific sentiment adjustments.

Sources: Google News RSS (free, no API key)
Topics: RBI, US Fed, India GDP, Crude Oil, IT sector,
        Pharma tariffs, FII flows, Rupee, Banking NPA, Auto sector
"""

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import time
import re
from news_sentiment import score_headline

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── Macro topic definitions ───────────────────────────────────────────────────
# Each topic has:
# - query: what to search on Google News
# - affects: list of stock symbols OR 'all'
# - weight: how strongly this topic affects the stocks (0.0 to 1.0)
# - direction: some topics are inverted (e.g. high crude = bad for OMCs)

MACRO_TOPICS = {

    'rbi_rate': {
        'query': 'RBI interest rate India monetary policy 2026',
        'affects': [
            'HDFCBANK','ICICIBANK','SBIN','KOTAKBANK','AXISBANK',
            'BAJFINANCE','BAJAJFINSV','CHOLAFIN','MUTHOOTFIN',
            'MANAPPURAM','SHRIRAMFIN','BANKBARODA','PNB','CANBK'
        ],
        'weight': 0.95,
        'invert': False,
        'extra_positive': ['rate cut','repo cut','dovish','liquidity','accommodative'],
        'extra_negative': ['rate hike','repo hike','hawkish','tightening','inflation'],
    },

    'us_fed': {
        'query': 'US Federal Reserve rate decision FII India 2026',
        'affects': 'all',
        'weight': 0.90,
        'invert': False,
        'extra_positive': ['rate cut','pause','dovish','fii inflow','foreign buying'],
        'extra_negative': ['rate hike','hawkish','recession','fii outflow','selling'],
    },

    'india_gdp': {
        'query': 'India GDP growth economy 2026',
        'affects': 'all',
        'weight': 0.75,
        'invert': False,
        'extra_positive': ['growth','beats','strong economy','expansion','recovery'],
        'extra_negative': ['slowdown','contraction','weak','recession','decline'],
    },

    'crude_oil': {
        'query': 'crude oil price Brent WTI India 2026',
        'affects': ['BPCL','IOC','ONGC','RELIANCE','TATAPOWER'],
        'weight': 0.80,
        'invert': True,  # high crude = BAD for oil marketing companies
        'extra_positive': ['falls','drops','decline','low','cheap'],
        'extra_negative': ['surges','rises','spike','high','expensive','opec cut'],
    },

    'it_sector': {
        'query': 'India IT sector US tech spending outsourcing 2026',
        'affects': [
            'TCS','INFY','WIPRO','HCLTECH','TECHM',
            'LTIM','PERSISTENT','MPHASIS','COFORGE','KPITTECH'
        ],
        'weight': 0.70,
        'invert': False,
        'extra_positive': ['deal win','strong demand','hiring','ai opportunity','growth'],
        'extra_negative': ['visa','h1b','layoff','slowdown','tariff','budget cut','weak demand'],
    },

    'pharma_us': {
        'query': 'India pharma US FDA tariff drug approval 2026',
        'affects': [
            'SUNPHARMA','DRREDDY','CIPLA','LUPIN',
            'AUROPHARMA','DIVISLAB','TORNTPHARM','ALKEM'
        ],
        'weight': 0.70,
        'invert': False,
        'extra_positive': ['fda approval','exemption','waiver','deal','generic approval'],
        'extra_negative': ['tariff','ban','warning letter','import alert','penalty'],
    },

    'fii_flows': {
        'query': 'FII DII foreign investor India stock market flows 2026',
        'affects': 'all',
        'weight': 0.85,
        'invert': False,
        'extra_positive': ['fii buying','inflow','foreign buying','bullish india','positive flows'],
        'extra_negative': ['fii selling','outflow','foreign selling','bearish india','negative flows'],
    },

    'rupee': {
        'query': 'USD INR rupee dollar exchange rate India 2026',
        'affects': [
            'TCS','INFY','WIPRO','HCLTECH','TECHM',
            'SUNPHARMA','DRREDDY','CIPLA','LUPIN'
        ],
        'weight': 0.50,
        'invert': False,
        'extra_positive': ['rupee weakens','rupee falls','dollar strengthens'],
        'extra_negative': ['rupee strengthens','rupee gains','dollar weakens'],
    },

    'banking_stress': {
        'query': 'India banking NPA bad loans stress RBI 2026',
        'affects': [
            'HDFCBANK','ICICIBANK','SBIN','KOTAKBANK','AXISBANK',
            'BANKBARODA','PNB','CANBK','BAJFINANCE','CHOLAFIN'
        ],
        'weight': 0.70,
        'invert': False,
        'extra_positive': ['npa falls','recovery','clean balance','credit growth','loan growth'],
        'extra_negative': ['npa rises','bad loans','stress','default','restructuring'],
    },

    'auto_sector': {
        'query': 'India automobile sector sales EV production 2026',
        'affects': [
            'MARUTI','M&M','BAJAJ-AUTO','HEROMOTOCO',
            'EICHERMOT','TVSMOTOR','MOTHERSON','BALKRISIND','APOLLOTYRE'
        ],
        'weight': 0.65,
        'invert': False,
        'extra_positive': ['sales growth','record sales','ev launch','demand surge','export growth'],
        'extra_negative': ['sales decline','slowdown','recall','chip shortage','weak demand'],
    },

    'fmcg_rural': {
        'query': 'India rural consumption FMCG demand 2026',
        'affects': [
            'HINDUNILVR','ITC','NESTLEIND','BRITANNIA','TATACONSUM',
            'MARICO','DABUR','COLPAL','GODREJCP','EMAMILTD'
        ],
        'weight': 0.55,
        'invert': False,
        'extra_positive': ['rural demand','volume growth','consumption','recovery','premiumisation'],
        'extra_negative': ['rural stress','volume decline','inflation impact','weak consumption'],
    },

    'infra_govt': {
        'query': 'India infrastructure government spending capex budget 2026',
        'affects': [
            'LT','ULTRACEMCO','GRASIM','ADANIPORTS','ADANIENT',
            'SIEMENS','ABB','CUMMINSIND','HAVELLS','POWERGRID','NTPC'
        ],
        'weight': 0.65,
        'invert': False,
        'extra_positive': ['capex','government spending','infra push','order win','contract'],
        'extra_negative': ['budget cut','spending cut','delay','cancellation'],
    },

    'global_risk': {
        'query': 'global recession US China trade war geopolitical risk 2026',
        'affects': 'all',
        'weight': 0.80,
        'invert': False,
        'extra_positive': ['trade deal','resolution','recovery','stimulus'],
        'extra_negative': ['recession','trade war','tariff','geopolitical','war','conflict','sanction'],
    },

    'union_budget': {
        'query': 'India Union Budget 2026 allocation spending fiscal',
        'affects': 'all',
        'weight': 0.60,
        'invert': False,
        'extra_positive': [
            'capex increase', 'spending boost', 'tax cut', 'relief',
            'allocation increased', 'fiscal stimulus', 'infrastructure push',
            'income tax', 'standard deduction', 'rebate'
        ],
        'extra_negative': [
            'tax hike', 'fiscal deficit', 'divestment', 'cut allocation',
            'surcharge', 'cess increase', 'spending cut'
        ],
    },

    'defence_budget': {
        'query': 'India defence budget allocation HAL BEL military spending 2026',
        'affects': ['BEL', 'HAL', 'BHARTIARTL', 'LT', 'ADANIENT'],
        'weight': 0.45,
        'invert': False,
        'extra_positive': [
            'defence allocation', 'increased budget', 'indigenisation',
            'make in india', 'order win', 'contract'
        ],
        'extra_negative': [
            'budget cut', 'reduced allocation', 'import', 'delay'
        ],
    },

    'income_tax': {
        'query': 'India income tax budget slab relief middle class 2026',
        'affects': [
            'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'MARICO',
            'DABUR', 'COLPAL', 'TITAN', 'MARUTI', 'BAJFINANCE'
        ],
        'weight': 0.50,
        'invert': False,
        'extra_positive': [
            'tax cut', 'relief', 'rebate', 'exemption', 'disposable income',
            'consumption boost', 'standard deduction increased'
        ],
        'extra_negative': [
            'tax hike', 'surcharge', 'new tax', 'withdrawal'
        ],
    },

    'capex_budget': {
        'query': 'India capital expenditure infrastructure budget 2026 government',
        'affects': [
            'LT', 'ULTRACEMCO', 'GRASIM', 'SIEMENS', 'ABB',
            'CUMMINSIND', 'HAVELLS', 'POWERGRID', 'NTPC',
            'ADANIPORTS', 'ADANIENT'
        ],
        'weight': 0.60,
        'invert': False,
        'extra_positive': [
            'capex boost', 'infrastructure push', 'record spending',
            'roads', 'railways', 'ports', 'power', 'increased allocation'
        ],
        'extra_negative': [
            'capex cut', 'reduced spending', 'fiscal consolidation',
            'budget deficit', 'spending squeeze'
        ],
    },

    'health_budget': {
        'query': 'India health budget pharma medical spending 2026',
        'affects': [
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN',
            'APOLLOHOSP', 'DIVISLAB', 'TORNTPHARM', 'ALKEM'
        ],
        'weight': 0.40,
        'invert': False,
        'extra_positive': [
            'health allocation', 'increased budget', 'ayushman',
            'insurance', 'healthcare push', 'jan aushadhi'
        ],
        'extra_negative': [
            'budget cut', 'reduced allocation', 'price control'
        ],
    },

    'monsoon': {
        'query': 'India monsoon rainfall 2026 IMD forecast agriculture',
        'affects': [
            'HINDUNILVR', 'ITC', 'MARICO', 'DABUR', 'COLPAL',
            'GODREJCP', 'EMAMILTD', 'TATACONSUM', 'BRITANNIA',
            'MARUTI', 'HEROMOTOCO', 'BAJAJ-AUTO'
        ],
        'weight': 0.35,
        'invert': False,
        'extra_positive': [
            'normal monsoon', 'above normal', 'good rainfall',
            'adequate rain', 'bumper crop', 'kharif'
        ],
        'extra_negative': [
            'deficient monsoon', 'drought', 'below normal',
            'poor rainfall', 'el nino', 'crop failure'
        ],
    },

    'gst_collections': {
        'query': 'India GST collection revenue 2026',
        'affects': 'all',
        'weight': 0.60,
        'invert': False,
        'extra_positive': [
            'record gst', 'high collection', 'growth', 'buoyant',
            'strong revenue', 'exceeds target'
        ],
        'extra_negative': [
            'gst miss', 'low collection', 'decline', 'below target',
            'shortfall', 'weak revenue'
        ],
    },

    'china_slowdown': {
        'query': 'China economy slowdown PMI demand 2026',
        'affects': [
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL',
            'COALINDIA', 'ADANIPORTS'
        ],
        'weight': 0.50,
        'invert': True,  # China slowdown = less demand for metals = bad
        'extra_positive': ['china growth', 'stimulus', 'recovery', 'demand surge'],
        'extra_negative': ['china slowdown', 'contraction', 'property crisis', 'weak demand'],
    },

    'us_recession': {
        'query': 'US recession economy slowdown 2026',
        'affects': [
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM',
            'LTIM', 'PERSISTENT', 'MPHASIS', 'COFORGE',
            'SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN'
        ],
        'weight': 1.00,
        'invert': False,
        'extra_positive': ['soft landing', 'recovery', 'strong jobs', 'gdp growth'],
        'extra_negative': ['recession', 'slowdown', 'job cuts', 'gdp contraction', 'tariff'],
    },

    'dollar_index': {
        'query': 'US dollar index DXY strength India emerging markets 2026',
        'affects': 'all',
        'weight': 0.75,
        'invert': True,  # strong dollar = bad for emerging markets like India
        'extra_positive': ['dollar weakens', 'dxy falls', 'emerging market rally'],
        'extra_negative': ['dollar strengthens', 'dxy rises', 'capital outflow', 'fii selling'],
    },

    'iip_pmi': {
        'query': 'India IIP PMI manufacturing industrial production 2026',
        'affects': [
            'LT', 'SIEMENS', 'ABB', 'CUMMINSIND', 'HAVELLS',
            'TATASTEEL', 'JSWSTEEL', 'ULTRACEMCO', 'GRASIM',
            'MOTHERSON', 'BALKRISIND', 'APOLLOTYRE'
        ],
        'weight': 0.60,
        'invert': False,
        'extra_positive': ['pmi expansion', 'iip growth', 'manufacturing growth',
                           'above 50', 'record output'],
        'extra_negative': ['pmi contraction', 'iip decline', 'below 50',
                           'manufacturing slowdown', 'weak output'],
    },

    'geopolitical': {
        'query': 'India geopolitical war conflict tension 2026',
        'affects': 'all',
        'weight': 1.00,
        'invert': False,
        'extra_positive': ['peace', 'ceasefire', 'resolution', 'trade deal', 'diplomacy'],
        'extra_negative': ['war', 'conflict', 'tension', 'sanctions', 'attack',
                           'escalation', 'missile', 'border'],
    },

    'npa_banking': {
        'query': 'India NPA gross bad loans banking sector RBI 2026',
        'affects': [
            'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
            'BANKBARODA', 'PNB', 'CANBK', 'BAJFINANCE', 'CHOLAFIN'
        ],
        'weight': 0.85,
        'invert': False,
        'extra_positive': ['npa declines', 'asset quality improves', 'recovery',
                           'provision coverage', 'clean books'],
        'extra_negative': ['npa rises', 'bad loans increase', 'stress',
                           'restructuring', 'write-off'],
    },
}


def fetch_macro_news(topic_name: str, topic: dict, max_articles: int = 8) -> list:
    """Fetch news for a macro topic."""
    query = topic['query']
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
            score = score_headline(title)

            # Apply extra topic-specific keywords
            h = title.lower()
            for kw in topic.get('extra_positive', []):
                if kw in h:
                    score += 4
            for kw in topic.get('extra_negative', []):
                if kw in h:
                    score -= 4

            # Invert if needed (e.g. high crude = bad)
            if topic.get('invert'):
                score = -score

            score = max(-10, min(10, score))
            articles.append({'title': title, 'score': score})

        return articles

    except Exception:
        return []


def get_macro_sentiment() -> dict:
    """
    Fetch all macro topics and return per-topic sentiment scores.
    Returns dict: { topic_name: { score, label, articles } }
    """
    results = {}

    print(f"\n  Fetching macro/economic news ({len(MACRO_TOPICS)} topics)...")

    for topic_name, topic in MACRO_TOPICS.items():
        print(f"    -> {topic_name}...", end=' ', flush=True)
        articles = fetch_macro_news(topic_name, topic)

        if not articles:
            score = 0
        else:
            scores = [a['score'] for a in articles]
            avg = sum(scores) / len(scores)
            score = round(avg * 10, 1)
            score = max(-100, min(100, score))

        label = 'positive' if score > 10 else 'negative' if score < -10 else 'neutral'
        print(f"{score:+.0f} ({label})")

        results[topic_name] = {
            'score':    score,
            'label':    label,
            'weight':   topic['weight'],
            'affects':  topic['affects'],
            'articles': len(articles),
            'top':      sorted(articles, key=lambda x: abs(x['score']),
                               reverse=True)[:2],
        }

        time.sleep(1.0)  # Be respectful to Google News

    return results


def apply_macro_to_stock(symbol: str, macro_data: dict) -> dict:
    """
    Apply relevant macro sentiment to a specific stock.
    Returns adjusted macro sentiment score for the stock.
    """
    applicable_topics = []

    for topic_name, topic_data in macro_data.items():
        affects = topic_data['affects']
        weight  = topic_data['weight']
        score   = topic_data['score']

        if affects == 'all' or symbol in affects:
            applicable_topics.append({
                'topic':  topic_name,
                'score':  score,
                'label':  topic_data['label'],
                'weight': weight,
            })

    if not applicable_topics:
        return {
            'macro_score':  0,
            'macro_label':  'neutral',
            'topics_count': 0,
            'topics':       [],
        }

    # True weighted average
    total_weight = sum(t['weight'] for t in applicable_topics)
    weighted_sum = sum(t['score'] * t['weight'] for t in applicable_topics)
    macro_score  = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0
    macro_score  = max(-100, min(100, macro_score))
    macro_label = 'positive' if macro_score > 10 else 'negative' if macro_score < -10 else 'neutral'

    return {
        'macro_score':  macro_score,
        'macro_label':  macro_label,
        'topics_count': len(applicable_topics),
        'topics':       applicable_topics[:3],
    }


if __name__ == "__main__":
    print("="*60)
    print("  Macro Sentiment Engine — Test Run")
    print("="*60)

    macro_data = get_macro_sentiment()

    print("\n" + "="*60)
    print("  Macro Impact per Sector")
    print("="*60)

    test_stocks = {
        'IT':      'TCS',
        'Banking': 'HDFCBANK',
        'Pharma':  'SUNPHARMA',
        'Auto':    'MARUTI',
        'Energy':  'BPCL',
        'FMCG':    'HINDUNILVR',
        'Infra':   'LT',
    }

    for sector, sym in test_stocks.items():
        result = apply_macro_to_stock(sym, macro_data)
        print(f"\n  {sym} ({sector}):")
        print(f"    Macro Score: {result['macro_score']:+.1f} — {result['macro_label'].upper()}")
        print(f"    Relevant topics: {result['topics_count']}")
        for t in result['topics']:
            icon = 'POS' if t['label'] == 'positive' else 'NEG' if t['label'] == 'negative' else 'NEU'
            print(f"      [{icon}] {t['topic']}: {t['score']:+.0f}")
