"""
Investment Report Generator
============================
Generates a professional PDF investment report for any NSE stock.
Called by /generate-report endpoint in api_server.py.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Color palette ─────────────────────────────────────────────────────────────
C_BG       = colors.HexColor('#0d1f3c')
C_SURFACE  = colors.HexColor('#1a2b4a')
C_GREEN    = colors.HexColor('#00c864')
C_GOLD     = colors.HexColor('#f59e0b')
C_RED      = colors.HexColor('#ef4444')
C_BLUE     = colors.HexColor('#3b82f6')
C_TEXT     = colors.HexColor('#e2e8f0')
C_MUTED    = colors.HexColor('#94a3b8')
C_WHITE    = colors.white
C_BORDER   = colors.HexColor('#2d3f5e')
C_ACCENT   = colors.HexColor('#f97316')

W, H = A4  # 210 x 297 mm


def verdict_color(verdict):
    v = (verdict or '').upper()
    if 'BUY' in v:   return C_GREEN
    if 'SELL' in v:  return C_RED
    return C_GOLD


def score_color(score):
    s = float(score or 0)
    if s >= 65: return C_GREEN
    if s >= 50: return C_GOLD
    return C_RED


def fmt_price(v):
    if v is None: return '—'
    try:
        return f"Rs.{float(v):,.2f}"
    except Exception:
        return str(v)


def fmt_pct(v, decimals=1):
    if v is None: return '—'
    try:
        return f"{float(v):.{decimals}f}%"
    except Exception:
        return str(v)


def fmt_num(v, decimals=1):
    if v is None: return '—'
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def safe(d, *keys, default='—'):
    for k in keys:
        if d is None: return default
        d = d.get(k) if isinstance(d, dict) else default
        if d is None: return default
    return d if d not in (None, '', 'None') else default


# ── Style factory ─────────────────────────────────────────────────────────────
def S(name, **kwargs):
    base = {
        'fontName':  'Helvetica',
        'fontSize':  10,
        'textColor': C_TEXT,
        'leading':   14,
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


# ── Section header ─────────────────────────────────────────────────────────────
def section_header(title):
    _p = Paragraph(title, S('sh', fontName='Helvetica-Bold', fontSize=9,
                   textColor=C_ACCENT, letterSpacing=1.5))
    t = Table([[_p]], colWidths=[W - 40*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_SURFACE),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('LINEBELOW',    (0,0), (-1,-1), 1.5, C_ACCENT),
    ]))
    return [Spacer(1, 3*mm), t, Spacer(1, 2*mm)]


def kv_table(rows, col_widths=None):
    """Two-column key-value table."""
    cw = col_widths or [(W-40*mm)*0.42, (W-40*mm)*0.58]
    data = []
    for k, v, *rest in rows:
        color = rest[0] if rest else C_TEXT
        data.append([
            Paragraph(str(k), S('k', textColor=C_MUTED, fontSize=9)),
            Paragraph(str(v), S('v', textColor=color, fontSize=10,
                                fontName='Helvetica-Bold')),
        ])
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_SURFACE),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [C_SURFACE, colors.HexColor('#1e3050')]),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LINEBELOW',    (0,-1), (-1,-1), 0.5, C_BORDER),
    ]))
    return t


def four_col_table(rows):
    """Four-column metric table."""
    cw = [(W-40*mm)/4] * 4
    data = []
    for i in range(0, len(rows), 2):
        row_data = []
        for j in range(2):
            if i+j < len(rows):
                k, v, *rest = rows[i+j]
                color = rest[0] if rest else C_TEXT
                row_data += [
                    Paragraph(str(k), S('k4', textColor=C_MUTED, fontSize=8)),
                    Paragraph(str(v), S('v4', textColor=color, fontSize=10,
                                        fontName='Helvetica-Bold')),
                ]
            else:
                row_data += [Paragraph('', S('e')), Paragraph('', S('e'))]
        data.append(row_data)

    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_SURFACE),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [C_SURFACE, colors.HexColor('#1e3050')]),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
    ]))
    return t


def forecast_table(forecast):
    """1Y / 3Y / 5Y forecast table."""
    headers = ['Period', '1 Year', '3 Years', '5 Years']
    periods = ['1y', '3y', '5y']

    def fv(period, key):
        f = (forecast or {}).get(period, {})
        return f.get(key)

    cur = (forecast or {}).get('current_price')
    rows = [headers]
    for label, key, fmt_fn in [
        ('Price Target',     'price_target',      fmt_price),
        ('Upside from now',  '_upside',           lambda v: fmt_pct(v)),
        ('Rev CAGR',         'revenue_growth_pct', lambda v: fmt_pct(v)),
        ('Profit CAGR',      'profit_growth_pct',  lambda v: fmt_pct(v)),
        ('Outperform Prob',  'outperform_prob',    lambda v: fmt_pct(v)),
    ]:
        row = [Paragraph(label, S('fh', textColor=C_MUTED, fontSize=9))]
        for p in periods:
            f  = (forecast or {}).get(p, {})
            pt = f.get('price_target')
            if key == '_upside':
                v = round((pt - cur) / cur * 100, 1) if pt and cur else None
            else:
                v = f.get(key)
            color = C_TEXT
            if key == 'outperform_prob':
                try:
                    color = C_GREEN if float(v or 0) >= 55 else C_GOLD if float(v or 0) >= 50 else C_RED
                except Exception:
                    pass
            if key == '_upside':
                try:
                    color = C_GREEN if float(v or 0) > 0 else C_RED
                except Exception:
                    pass
            row.append(Paragraph(fmt_fn(v), S('fv', textColor=color,
                                              fontSize=10, fontName='Helvetica-Bold')))
        rows.append(row)

    cw = [50*mm] + [(W - 40*mm - 50*mm) / 3] * 3
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  C_BG),
        ('BACKGROUND',    (0,1), (-1,-1), C_SURFACE),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [C_SURFACE, colors.HexColor('#1e3050')]),
        ('TEXTCOLOR',     (0,0), (-1,0),  C_MUTED),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0),  9),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LINEBELOW',     (0,0), (-1,0),  1, C_BORDER),
        ('LINEBELOW',     (0,-1),(-1,-1), 0.5, C_BORDER),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
    ]))
    return t


def news_table(headlines):
    rows = []
    for h in (headlines or [])[:3]:
        score = h.get('sentiment_score', h.get('score', 0)) or 0
        label = h.get('sentiment', h.get('sentiment_label', 'neutral'))
        color = C_GREEN if label == 'positive' else C_RED if label == 'negative' else C_MUTED
        score_str = f"+{score}" if score > 0 else str(score)
        rows.append([
            Paragraph(h.get('title', '—'), S('nt', textColor=C_TEXT, fontSize=8, leading=12)),
            Paragraph(score_str, S('ns', textColor=color, fontSize=9,
                                   fontName='Helvetica-Bold', alignment=TA_RIGHT)),
        ])
    if not rows:
        rows = [[Paragraph('No recent headlines available.', S('nm', textColor=C_MUTED, fontSize=9)), Paragraph('', S('e'))]]

    cw = [(W-40*mm)*0.85, (W-40*mm)*0.15]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_SURFACE),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [C_SURFACE, colors.HexColor('#1e3050')]),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW',     (0,0), (-1,-2), 0.5, C_BORDER),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t


# ── Main generator ────────────────────────────────────────────────────────────
def generate_report(data: dict) -> bytes:
    """Generate PDF report and return as bytes."""
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
    )

    quote   = data.get('quote', {}) or {}
    ml      = data.get('ml', {}) or {}
    fund    = data.get('fundamentals', {}) or {}
    val     = data.get('valuation', {}) or {}
    sent    = data.get('sentiment', {}) or {}
    macro   = data.get('macro', {}) or {}
    combined= data.get('combined', {}) or {}
    forecast= data.get('forecast', {}) or {}
    vs      = combined.get('valuation_signal') or {}

    symbol   = data.get('symbol', '—')
    company  = safe(quote, 'company_name', default=symbol)
    sector   = safe(quote, 'industry', default='—')
    price    = safe(quote, 'price')
    verdict  = safe(combined, 'verdict', default='HOLD')
    score_10 = safe(combined, 'score_10', default='—')
    grade    = safe(combined, 'grade', default='—')
    risk     = safe(combined, 'risk', default='Medium')
    date_str = datetime.now().strftime('%d %B %Y, %I:%M %p IST')

    vc = verdict_color(verdict)
    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 20*mm))

    # Header banner
    story.append(Table([[
        Paragraph('GRAHAM', S('logo', fontName='Helvetica-Bold', fontSize=22,
                               textColor=C_ACCENT)),
        Paragraph('INDIA EQUITY SCREENER', S('sub', fontSize=9, textColor=C_MUTED,
                                              alignment=TA_RIGHT)),
    ]], colWidths=[(W-40*mm)*0.5, (W-40*mm)*0.5]))
    story[-1].setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_BG),
        ('LEFTPADDING',  (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('TOPPADDING',   (0,0), (-1,-1), 12),
        ('BOTTOMPADDING',(0,0), (-1,-1), 12),
        ('LINEBELOW',    (0,0), (-1,-1), 2, C_ACCENT),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))

    story.append(Spacer(1, 12*mm))

    # Stock name block
    story.append(Table([[
        Paragraph(company, S('cn', fontName='Helvetica-Bold', fontSize=26,
                              textColor=C_WHITE, leading=32)),
    ], [
        Paragraph(f'{symbol}   \u00b7   NSE   \u00b7   {sector}',
                  S('sym', fontSize=11, textColor=C_MUTED, leading=16)),
    ]], colWidths=[W-40*mm], rowHeights=[36, 20]))
    story[-1].setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_BG),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 0),
        ('TOPPADDING',    (0,0), (0,0),   4),
        ('BOTTOMPADDING', (0,0), (0,0),   8),
        ('TOPPADDING',    (0,1), (0,1),   0),
        ('BOTTOMPADDING', (0,1), (0,1),   4),
    ]))
    story.append(Spacer(1, 4*mm))

    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 10*mm))

    # Key metrics row — flat 2-row table: labels on top, values below
    price_str = fmt_price(price)
    cw4 = [(W-40*mm)/4] * 4
    metrics_labels = Table([[
        Paragraph('CURRENT PRICE', S('ml', textColor=C_MUTED, fontSize=8)),
        Paragraph('VERDICT',       S('ml', textColor=C_MUTED, fontSize=8)),
        Paragraph('SCORE',         S('ml', textColor=C_MUTED, fontSize=8)),
        Paragraph('GRADE',         S('ml', textColor=C_MUTED, fontSize=8)),
    ]], colWidths=cw4)
    metrics_labels.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_SURFACE),
        ('LEFTPADDING',  (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING',   (0,0), (-1,-1), 12),
        ('BOTTOMPADDING',(0,0), (-1,-1), 2),
        ('LINEAFTER',    (0,0), (-2,-1), 0.5, C_BORDER),
    ]))
    metrics_values = Table([[
        Paragraph(str(price_str), S('mv', fontName='Helvetica-Bold', fontSize=18, textColor=C_WHITE)),
        Paragraph(str(verdict),   S('mv', fontName='Helvetica-Bold', fontSize=18, textColor=vc)),
        Paragraph(f"{score_10}/10", S('mv', fontName='Helvetica-Bold', fontSize=18,
                                      textColor=score_color(combined.get('score', 50)))),
        Paragraph(str(grade),     S('mv', fontName='Helvetica-Bold', fontSize=18, textColor=C_GOLD)),
    ]], colWidths=cw4)
    metrics_values.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), C_SURFACE),
        ('LEFTPADDING',  (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING',   (0,0), (-1,-1), 2),
        ('BOTTOMPADDING',(0,0), (-1,-1), 14),
        ('LINEAFTER',    (0,0), (-2,-1), 0.5, C_BORDER),
    ]))
    story.append(metrics_labels)
    story.append(metrics_values)

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(f'Report generated: {date_str}',
                            S('date', fontSize=8, textColor=C_MUTED)))

    # ══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('EXECUTIVE SUMMARY')

    reason    = safe(combined, 'reason', default='Mixed signals')
    ml_score  = safe(ml, 'ml_score')
    prediction= safe(ml, 'prediction', default='—')
    risk_lvl  = safe(combined, 'risk', default='Medium')
    pt_1y     = safe(forecast, '1y', 'price_target')
    upside    = None
    if pt_1y and pt_1y != '—' and price:
        try:
            upside = round((float(pt_1y) - float(price)) / float(price) * 100, 1)
        except Exception:
            pass

    upside_str = f" with a 1-year price target of {fmt_price(pt_1y)} ({'+' if upside and upside > 0 else ''}{upside}% upside)" if pt_1y and pt_1y != '—' else ""
    summary_text = (
        f"<b>{company}</b> ({symbol}) is currently rated <b>{verdict}</b> with a score of "
        f"<b>{score_10}/10</b>. {reason.capitalize()}. "
        f"The ML model ({safe(ml, 'accuracy')}% accuracy) predicts the stock will "
        f"<b>{prediction.lower()}</b> the Nifty 50 over the next 3 months{upside_str}. "
        f"Risk level is assessed as <b>{risk_lvl}</b>."
    )
    story.append(Paragraph(summary_text, S('sum', fontSize=10, leading=16,
                                            textColor=C_TEXT)))

    # Valuation signal box
    if vs:
        story.append(Spacer(1, 4*mm))
        sig_color = C_GREEN if vs.get('color') == 'green' else C_RED if vs.get('color') == 'red' else C_GOLD
        story.append(Table([[
            Paragraph(f"Valuation Signal: <b>{vs.get('label','—')}</b>  —  {vs.get('description','')}", 
                      S('vs', fontSize=9, textColor=sig_color)),
        ]], colWidths=[W-40*mm]))
        story[-1].setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_BG),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('TOPPADDING',    (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 9),
            ('LINEBEFORE',    (0,0), (0,-1),  3, sig_color),
        ]))

    # ══════════════════════════════════════════════════════════════════════════
    # FORWARD FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('FORWARD FORECAST')
    story.append(forecast_table(forecast))

    if vs and vs.get('fair_value'):
        story.append(Spacer(1, 3*mm))
        story.append(kv_table([
            ('Graham Fair Value', fmt_price(vs.get('fair_value')), C_GOLD),
            ('Buy Zone',
             f"{fmt_price(vs.get('buy_zone_low'))} – {fmt_price(vs.get('buy_zone_high'))}",
             C_GREEN),
            ('Confidence', fmt_pct(vs.get('confidence')), C_TEXT),
            ('vs Fair Value', f"{'+' if (vs.get('pct_vs_fair') or 0) > 0 else ''}{fmt_num(vs.get('pct_vs_fair'))}%",
             C_GREEN if (vs.get('pct_vs_fair') or 0) < 0 else C_RED),
        ]))

    # ══════════════════════════════════════════════════════════════════════════
    # 10-YEAR FUNDAMENTALS
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('10-YEAR FUNDAMENTALS')

    fcf_ok   = fund.get('fcf_positive_3y')
    debt_red = fund.get('debt_reducing')
    story.append(four_col_table([
        ('ROCE',           fmt_pct(fund.get('roce')),          score_color(fund.get('roce', 0))),
        ('Sales CAGR 5Y',  fmt_pct(fund.get('sales_cagr_5y')), C_TEXT),
        ('Profit CAGR 5Y', fmt_pct(fund.get('profit_cagr_5y')),C_TEXT),
        ('EPS CAGR 5Y',    fmt_pct(fund.get('eps_cagr_5y')),   C_TEXT),
        ('Promoter %',     fmt_pct(fund.get('promoter_pct')),  C_TEXT),
        ('OPM Latest',     fmt_pct(fund.get('opm_latest_pct')),C_TEXT),
        ('FCF Positive 3Y','Yes' if fcf_ok else 'No',          C_GREEN if fcf_ok else C_RED),
        ('Debt Reducing',  'Yes' if debt_red else 'No',        C_GREEN if debt_red else C_RED),
        ('Investment Score',str(fund.get('investment_score','—')), score_color(fund.get('investment_score', 50))),
        ('Investment Grade',str(fund.get('investment_grade','—')), C_GOLD),
    ]))

    # ══════════════════════════════════════════════════════════════════════════
    # VALUATION METRICS
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('VALUATION METRICS')

    mcap     = safe(quote, 'market_cap', default='—')
    w52_high = safe(quote, 'week52_high')
    w52_low  = safe(quote, 'week52_low')
    pos52    = safe(ml, 'pos52_pct')
    div_y    = val.get('dividend_yield')
    div_str  = f"{float(div_y)*100:.2f}%" if div_y else '—'

    story.append(four_col_table([
        ('P/E Ratio',    fmt_num(val.get('pe_ratio')),  C_TEXT),
        ('EPS',          fmt_price(val.get('eps')),      C_TEXT),
        ('Dividend Yield',div_str,                       C_TEXT),
        ('Market Cap',   str(mcap).replace('₹','Rs.').replace('\u20b9','Rs.'), C_TEXT),
        ('52W High',     fmt_price(w52_high),             C_RED),
        ('52W Low',      fmt_price(w52_low),              C_GREEN),
        ('52W Position', fmt_pct(pos52),                  C_TEXT),
        ('RSI (14)',     fmt_num(ml.get('rsi')),          C_TEXT),
    ]))

    # ══════════════════════════════════════════════════════════════════════════
    # ML ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('ML ANALYSIS')

    story.append(four_col_table([
        ('ML Score',        fmt_pct(ml.get('ml_score')),       score_color(ml.get('ml_score', 50))),
        ('ML Prediction',   safe(ml, 'prediction', default='—'), C_GREEN if prediction == 'OUTPERFORM' else C_RED),
        ('Combined Score',  fmt_pct(combined.get('score')),     score_color(combined.get('score', 50))),
        ('Model Accuracy',  fmt_pct(ml.get('accuracy')),        C_TEXT),
        ('Screener Score',  fmt_num(combined.get('screener_score')), C_TEXT),
        ('News Score',      fmt_num(combined.get('sent_score')),C_TEXT),
        ('Macro Score',     fmt_num(combined.get('macro_score')),C_TEXT),
        ('1M Return',       f"{'+' if (ml.get('ret_1m_pct') or 0) > 0 else ''}{fmt_num(ml.get('ret_1m_pct'))}%",
         C_GREEN if (ml.get('ret_1m_pct') or 0) > 0 else C_RED),
    ]))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        'Scoring weights: 35% ML · 20% Fundamentals · 15% Valuation · 15% News · 15% Macro',
        S('wt', fontSize=8, textColor=C_MUTED)))

    # ══════════════════════════════════════════════════════════════════════════
    # NEWS SENTIMENT
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('NEWS SENTIMENT')

    headlines = sent.get('top_headlines', [])
    sent_score= safe(sent, 'sentiment_score', default='0')
    sent_label= safe(sent, 'sentiment_label', default='neutral')
    sc = C_GREEN if sent_label == 'positive' else C_RED if sent_label == 'negative' else C_MUTED

    story.append(Table([[
        Paragraph(f"Overall sentiment: <b>{sent_label.upper()}</b>  ({'+' if float(sent_score or 0) > 0 else ''}{sent_score})",
                  S('sl', fontSize=9, textColor=sc)),
        Paragraph(f"Articles analysed: {safe(sent, 'total_articles', default='—')}",
                  S('sa', fontSize=8, textColor=C_MUTED, alignment=TA_RIGHT)),
    ]], colWidths=[(W-40*mm)*0.6, (W-40*mm)*0.4]))
    story[-1].setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),
                                    ('RIGHTPADDING',(0,0),(-1,-1),0),
                                    ('TOPPADDING',(0,0),(-1,-1),0),
                                    ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(news_table(headlines))

    # ══════════════════════════════════════════════════════════════════════════
    # MACRO FACTORS
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('MACRO & ECONOMIC FACTORS')

    macro_label = safe(macro, 'macro_label', default='neutral')
    macro_score = safe(macro, 'macro_score', default='0')
    mc = C_GREEN if macro_label == 'positive' else C_RED if macro_label == 'negative' else C_MUTED
    story.append(Paragraph(
        f"Overall macro: <b>{macro_label.upper()}</b>  ({'+' if float(macro_score or 0) > 0 else ''}{macro_score})",
        S('ml2', fontSize=9, textColor=mc)))
    story.append(Spacer(1, 3*mm))

    topics = (macro.get('topics') or [])[:3]
    if topics:
        topic_rows = []
        for t in topics:
            sc2 = t.get('score', 0) or 0
            lbl = t.get('label', 'neutral')
            col = C_GREEN if lbl == 'positive' else C_RED if lbl == 'negative' else C_MUTED
            topic_rows.append([
                Paragraph(str(t.get('topic', '—')).replace('_', ' ').title(),
                          S('tt', textColor=C_TEXT, fontSize=9)),
                Paragraph(lbl.capitalize(), S('tl', textColor=col, fontSize=9)),
                Paragraph(f"{'+' if sc2 > 0 else ''}{sc2}",
                          S('ts', textColor=col, fontSize=9, fontName='Helvetica-Bold',
                             alignment=TA_RIGHT)),
            ])
        mt = Table(topic_rows, colWidths=[(W-40*mm)*0.5, (W-40*mm)*0.3, (W-40*mm)*0.2])
        mt.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_SURFACE),
            ('ROWBACKGROUNDS',(0,0), (-1,-1), [C_SURFACE, colors.HexColor('#1e3050')]),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LINEBELOW',     (0,0), (-1,-2), 0.5, C_BORDER),
        ]))
        story.append(mt)

    # ══════════════════════════════════════════════════════════════════════════
    # DISCLAIMER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        '<b>Risk Disclaimer:</b> This report is generated by Graham India Equity Screener for '
        '<b>educational purposes only</b> and does not constitute financial advice. Past performance '
        'and ML predictions are not indicative of future results. Price targets are projections based '
        'on historical data and may not be achieved. Always consult a SEBI-registered investment '
        'advisor before making investment decisions. The authors accept no liability for investment '
        'decisions made based on this report.',
        S('disc', fontSize=7.5, textColor=colors.HexColor('#64748b'), leading=11)))

    # ── Build ─────────────────────────────────────────────────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        # Dark page background
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Footer bar
        canvas.setFillColor(C_SURFACE)
        canvas.rect(0, 0, W, 16*mm, fill=1, stroke=0)
        canvas.setFillColor(C_MUTED)
        canvas.setFont('Helvetica', 7)
        canvas.drawString(20*mm, 6*mm, f'Graham India Equity Screener  ·  {symbol}  ·  {date_str}')
        canvas.drawRightString(W - 20*mm, 6*mm, f'Page {doc.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf.read()
