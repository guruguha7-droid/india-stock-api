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
    score_10 = safe(combined, 'score', default=safe(combined, 'score_10', default='—'))
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
        Paragraph(f"{score_10}/100", S('mv', fontName='Helvetica-Bold', fontSize=18,
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
        f"<b>{score_10}/100</b>. {reason.capitalize()}. "
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
    # WHAT WOULD GRAHAM PAY?
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header('WHAT WOULD GRAHAM PAY?')
    try:
        if vs:
            fair_v   = vs.get('fair_value')
            buy_lo   = vs.get('buy_zone_low')
            buy_hi   = vs.get('buy_zone_high')
            pct_f    = float(vs.get('pct_vs_fair') or 0)
            conf     = vs.get('confidence')
            cur_p    = float(vs.get('current_price') or price or 0)
            fair_pe  = vs.get('fair_pe')
            is_below = pct_f < 0
            is_in    = buy_lo and buy_hi and cur_p >= float(buy_lo) and cur_p <= float(buy_hi)
            is_belowz= buy_lo and cur_p < float(buy_lo)
            sig_c    = C_GREEN if is_below else C_RED if pct_f > 15 else C_GOLD

            def _p(v):
                try: return f"Rs.{float(v):,.0f}"
                except: return '\u2014'

            if is_belowz:
                headline = (f"Graham's updated model estimates the fair value of <b>{company}</b> at "
                           f"<b>{_p(fair_v)}</b> (Fair P/E: {fair_pe}x). "
                           f"At the current price of <b>{_p(cur_p)}</b>, the stock trades at a "
                           f"<b>{abs(pct_f):.1f}% discount</b> to fair value \u2014 "
                           f"already below the buy zone of {_p(buy_lo)}\u2013{_p(buy_hi)}. "
                           f"This is a better entry than the target range itself.")
            elif is_in:
                headline = (f"Graham's updated model estimates the fair value of <b>{company}</b> at "
                           f"<b>{_p(fair_v)}</b> (Fair P/E: {fair_pe}x). "
                           f"The current price of <b>{_p(cur_p)}</b> sits right inside the buy zone "
                           f"of {_p(buy_lo)}\u2013{_p(buy_hi)} \u2014 a fair entry point with "
                           f"{conf}% model confidence.")
            elif pct_f > 0:
                headline = (f"Graham's updated model estimates the fair value of <b>{company}</b> at "
                           f"<b>{_p(fair_v)}</b> (Fair P/E: {fair_pe}x). "
                           f"At {_p(cur_p)}, the stock trades <b>{pct_f:.1f}% above</b> fair value. "
                           f"Wait for a pullback to the buy zone of {_p(buy_lo)}\u2013{_p(buy_hi)} "
                           f"before entering.")
            else:
                headline = (f"Graham's updated model estimates the fair value of <b>{company}</b> at "
                           f"<b>{_p(fair_v)}</b> (Fair P/E: {fair_pe}x). "
                           f"Current price is near fair value.")

            story.append(Paragraph(headline, S('gh', fontSize=11, textColor=C_TEXT, leading=16)))
            story.append(Spacer(1, 4*mm))

            story.append(kv_table([
                ('Graham Fair Value',  _p(fair_v),   C_GOLD),
                ('Fair P/E Used',      f"{fair_pe}x" if fair_pe else '\u2014', C_TEXT),
                ('Buy Zone',           f"{_p(buy_lo)} \u2013 {_p(buy_hi)}", C_GREEN),
                ('Current Price',      _p(cur_p),    sig_c),
                ('Discount / Premium', f"{'+' if pct_f >= 0 else ''}{pct_f:.1f}%", sig_c),
                ('Model Confidence',   f"{conf}%" if conf else '\u2014', C_TEXT),
                ('Valuation Signal',   vs.get('label','\u2014'), sig_c),
            ]))
        else:
            story.append(Paragraph(
                'Valuation data unavailable for this stock.',
                S('gna', fontSize=10, textColor=C_MUTED)))
    except Exception:
        pass

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
        'Scoring weights: 22% ML · 44% Fundamentals · 34% Valuation (sentiment gated)',
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
    # GRAHAM'S ADVICE
    # ══════════════════════════════════════════════════════════════════════════
    story += section_header("GRAHAM'S ADVICE")
    try:
        import hashlib, random as _rnd
        # Deterministic seed per symbol — same stock = same template, different stocks = different phrasing
        _seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        _rnd.seed(_seed)
        def pick(options): return _rnd.choice(options)

        vs_label  = vs.get('label', '') if vs else ''
        fair_val  = vs.get('fair_value') if vs else None
        buy_low   = vs.get('buy_zone_low') if vs else None
        buy_high  = vs.get('buy_zone_high') if vs else None
        pct_fair  = float(vs.get('pct_vs_fair') or 0) if vs else 0
        cur_price = float(vs.get('current_price') or price or 0)
        confidence= int(vs.get('confidence') or 0) if vs else 0

        scr      = float(safe(fund, 'investment_score', default=50))
        ml_s     = float(safe(ml, 'ml_score', default=50))
        roce_v   = float(safe(fund, 'roce', default=10))
        promoter = float(safe(fund, 'promoter_pct', default=40))
        fcf_ok   = fund.get('fcf_positive_3y')
        debt_red = fund.get('debt_reducing')
        sent_lbl = safe(sent, 'sentiment_label', default='neutral')
        news_s   = float(safe(sent, 'sentiment_score', default=0))
        pt_1y    = safe(forecast, '1y', 'price_target')
        eps_cagr = float(safe(fund, 'eps_cagr_5y', default=8))
        profit_c = float(safe(fund, 'profit_cagr_5y', default=8))
        qual     = 'strong' if scr >= 70 else 'moderate' if scr >= 50 else 'weak'
        inv_type = 'conservative and moderate' if scr >= 75 and risk != 'High' else 'moderate and aggressive' if scr >= 55 else 'aggressive'

        def _p(v):
            try: return f"{float(v):,.0f}"
            except: return str(v) if v and v != '\u2014' else '\u2014'

        # ── P1: Overall verdict ───────────────────────────────────────
        if verdict in ('BUY', 'STRONG BUY') and pct_fair < -10:
            p1 = pick([
                f"{company} stands out as a compelling buy right now \u2014 trading {abs(pct_fair):.0f}% below its Graham fair value of Rs.{_p(fair_val)}, the market appears to be mispricing this business. With {qual} fundamentals and an ML outperform probability of {ml_s:.0f}%, the risk-reward here is attractive.",
                f"At Rs.{_p(cur_price)}, {company} is trading well below its estimated fair value of Rs.{_p(fair_val)} \u2014 a {abs(pct_fair):.0f}% discount that rarely lasts long for a business of this quality. The combined score of {score_10}/100 and positive ML signal make this one worth acting on.",
                f"This is the kind of setup value investors look for \u2014 {company} has {qual} fundamentals (score {score_10}/100) but is priced {abs(pct_fair):.0f}% below fair value at Rs.{_p(cur_price)}. When quality meets discount, it usually pays to pay attention.",
            ])
        elif verdict in ('BUY', 'STRONG BUY'):
            _val_note = 'The valuation is reasonable near fair value.' if abs(pct_fair) < 10 else f'At {abs(pct_fair):.0f}% above fair value, patience at entry will improve your returns.'
            _val_note2 = 'Valuation is fair.' if abs(pct_fair) < 10 else f'The stock trades {pct_fair:.0f}% above fair value so a staggered entry is wise.'
            p1 = pick([
                f"{company} earns a BUY rating with a score of {score_10}/100. The business has delivered {profit_c:.0f}% profit CAGR over 5 years and the ML model gives it a {ml_s:.0f}% probability of outperforming Nifty. {_val_note}",
                f"With {qual} fundamentals and a positive ML signal, {company} makes a solid case for a BUY at current levels. A {profit_c:.0f}% profit CAGR and ROCE of {roce_v:.0f}% suggest the business is growing efficiently. {_val_note2}",
                f"{company} scores {score_10}/100 and is rated BUY. The {qual} fundamentals \u2014 {profit_c:.0f}% profit CAGR, {roce_v:.0f}% ROCE \u2014 give this business staying power. The ML model backs the thesis with a {ml_s:.0f}% outperform probability.",
            ])
        elif verdict == 'MILD BUY':
            p1 = pick([
                f"{company} earns a MILD BUY — the fundamentals are solid (score {score_10}/100) but the entry point isn't perfect yet. Consider starting with a small position and adding on dips.",
                f"A MILD BUY on {company} means the business is good but conviction isn't high enough for a full position at current levels. Start with 50% of your intended allocation and wait for a better price.",
                f"{company} scores {score_10}/100 — good enough to initiate a small position but not enough to go all-in. Build gradually rather than committing fully at current levels.",
            ])
        elif verdict == 'HOLD':
            _hdir    = 'above' if pct_fair > 0 else 'below'
            _vs_fair = f"{abs(pct_fair):.0f}% {_hdir} its Graham fair value of Rs.{_p(fair_val)}" if abs(pct_fair) >= 10 else "near its Graham fair value"
            _action  = "new investors can consider a small position here given the margin of safety" if pct_fair < -10 else "new investors are better off waiting for a pullback"
            p1 = pick([
                f"{company} is a HOLD \u2014 {qual} business (score {score_10}/100) currently trading {_vs_fair}. Existing holders have no reason to exit; {_action}.",
                f"The HOLD rating on {company} reflects a {qual} business at {'a stretched' if pct_fair > 15 else 'a fair' if abs(pct_fair) < 10 else 'an attractive'} valuation. Score of {score_10}/100 means it is worth keeping in the portfolio but not necessarily adding aggressively right now.",
                f"{company} sits comfortably in HOLD territory \u2014 {qual} fundamentals with a score of {score_10}/100, trading {_vs_fair}. Hold what you have; {_action}.",
            ])
        else:
            _sell_lead = 'Weak fundamentals combined with ' if scr < 50 else ''
            _sell_sent = 'negative news sentiment' if news_s < -10 else 'unfavourable ML signals'
            p1 = pick([
                f"{company} is rated SELL at current levels \u2014 a score of {score_10}/100 signals that the risk-reward does not favour buyers right now. {_sell_lead.capitalize()}{_sell_sent} suggest looking elsewhere.",
                f"The SELL rating on {company} is driven by a combination of {qual} fundamentals and unfavourable technicals. At a score of {score_10}/100, there are better opportunities in the market than holding or buying here.",
                f"{company} scores {score_10}/100 \u2014 not strong enough to justify buying or holding at current levels. Until fundamentals improve or the price corrects significantly, this one is best avoided.",
            ])

        # ── P2: Who should buy and at what price ──────────────────────
        _vol_note = "high recent volatility" if abs(float(safe(ml, 'ret_1m_pct', default=0))) > 8 else "near-term uncertainty"
        if buy_low and buy_high and cur_price:
            if cur_price < float(buy_low):
                _upside = round((float(buy_high) - cur_price) / cur_price * 100, 0) if buy_high else 0
                _below  = round((float(buy_low) - cur_price) / cur_price * 100, 0)
                p2 = pick([
                    f"At Rs.{_p(cur_price)}, the stock sits {_below:.0f}% below the buy zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} \u2014 meaning you can enter at an even better price than our target range. {inv_type.capitalize()} investors can start building a position now, with a target of Rs.{_p(fair_val)} implying {_upside:.0f}%+ upside.",
                    f"The current price of Rs.{_p(cur_price)} is actually cheaper than the buy zone (Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)}), which is the best possible entry scenario. {inv_type.capitalize()} investors should take advantage \u2014 a position here with {confidence}% confidence in fair value of Rs.{_p(fair_val)} makes strong mathematical sense.",
                    f"Rarely does a stock trade below its own buy zone \u2014 {company} at Rs.{_p(cur_price)} is doing exactly that. The buy zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} was already considered attractive; current levels are even more so. Suitable for {inv_type} investors willing to be patient.",
                ])
            elif cur_price <= float(buy_high):
                p2 = pick([
                    f"The stock is right inside the buy zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} \u2014 this is the ideal entry range. A staggered buy across 2\u20133 tranches is recommended to average out {_vol_note}. Suitable for {inv_type} investors.",
                    f"At Rs.{_p(cur_price)}, you are buying within the target zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)}. This is where the margin of safety is strongest. {inv_type.capitalize()} investors should consider initiating a position, ideally split across 2 tranches.",
                    f"Current price falls within the buy zone \u2014 {inv_type} investors are well-positioned to enter here. A disciplined entry between Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} with {confidence}% confidence is a sound approach.",
                ])
            else:
                _premium = abs(round(pct_fair, 0))
                _caution = 'Only aggressive investors should consider entering now.' if scr >= 70 else 'Avoid until the price corrects.'
                p2 = pick([
                    f"The stock trades {_premium:.0f}% above fair value at Rs.{_p(cur_price)} \u2014 patience here will be rewarded. Wait for a pullback to the buy zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} before committing capital. {_caution}",
                    f"At {_premium:.0f}% premium to fair value, the risk-reward is not compelling right now. The buy zone of Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)} offers a much better entry. Set a price alert and wait.",
                    f"Chasing {company} at Rs.{_p(cur_price)} means paying {_premium:.0f}% above what the fundamentals justify. The smarter move is to wait for Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)}.",
                ])
        else:
            _fund_note = f"A ROCE of {roce_v:.0f}% and {promoter:.0f}% promoter holding point to a business run with discipline." if scr >= 70 else "Position sizing conservatively is advised given the moderate fundamentals."
            p2 = pick([
                f"This stock suits {inv_type} investors. {_fund_note}",
                f"{inv_type.capitalize()} investors are the right audience for {company}. {'Strong ROCE and high promoter ownership suggest alignment between management and shareholders.' if scr >= 70 else 'Keep position size small until the business shows clearer improvement.'}",
            ])

        # ── P3: Key risks ─────────────────────────────────────────────
        risks = []
        if pct_fair > 20:          risks.append(f"the stock is priced {pct_fair:.0f}% above fair value, leaving little room for error")
        if profit_c < 5:           risks.append("profit growth has been underwhelming")
        if not fcf_ok:             risks.append("free cash flow has been inconsistent \u2014 the business is not generating reliable surplus cash")
        if not debt_red:           risks.append("debt levels are not declining, which adds financial risk")
        if promoter < 35:          risks.append(f"promoter holding of just {promoter:.0f}% raises questions about management confidence")
        if news_s < -10:           risks.append("recent news flow has been negative")
        if float(safe(ml, 'ret_3m_pct', default=0)) < -10: risks.append("price has been weak over the past 3 months, suggesting selling pressure")
        if sent_lbl == 'negative': risks.append("market sentiment is currently working against this stock")

        if risks:
            _r_intros = ["The main risks here are worth noting:", "Before investing, consider these risks:", "A few things to watch carefully:"]
            _r_outros = ["As with all investments, position size matters \u2014 never over-allocate to a single stock.", "Always size your position according to your personal risk tolerance.", "Keep stops in mind and do not invest money you cannot afford to lose."]
            risk_str = risks[0].capitalize()
            if len(risks) > 1: risk_str += f"; {risks[1]}"
            if len(risks) > 2: risk_str += f"; and {risks[2]}"
            p3 = f"{pick(_r_intros)} {risk_str}. {pick(_r_outros)}"
        else:
            _debt_note = 'is reducing' if debt_red else 'is stable'
            _fcf_note  = 'positive FCF' if fcf_ok else 'stable operations'
            p3 = pick([
                f"No major red flags are visible \u2014 {qual} fundamentals, {'positive' if fcf_ok else 'stable'} cash flow, and {promoter:.0f}% promoter holding suggest a well-run business. Standard market and sector risks still apply.",
                f"The risk profile here is relatively clean \u2014 debt {_debt_note}, promoter holding is at {promoter:.0f}%, and no major governance concerns are visible. Keep an eye on macro headwinds affecting the sector.",
                f"Fundamentals are solid with no glaring risks at this time. ROCE of {roce_v:.0f}% and {_fcf_note} suggest financial discipline. Monitor quarterly results for any deterioration.",
            ])

        # ── P4: Bottom line ───────────────────────────────────────────
        if verdict in ('BUY', 'STRONG BUY') and pct_fair < -5:
            _pt = f"with a 1-year target of Rs.{_p(pt_1y)}" if pt_1y and pt_1y != '\u2014' else "for long-term wealth creation"
            p4 = pick([
                f"Bottom line: {company} is a quality business on sale \u2014 accumulate below Rs.{_p(buy_high)} {_pt}.",
                f"Bottom line: The combination of strong fundamentals and attractive valuation makes {company} a priority buy. Start a position now and add on further dips {_pt}.",
                f"Bottom line: Don't overthink this one \u2014 {company} below Rs.{_p(buy_high)} is a solid bet {_pt}.",
            ])
        elif verdict in ('BUY', 'STRONG BUY'):
            _bz = f"on a dip to Rs.{_p(buy_low)}\u2013Rs.{_p(buy_high)}" if buy_low else "on market dips"
            p4 = pick([
                f"Bottom line: {company} is worth owning \u2014 enter {_bz} for the best risk-reward.",
                f"Bottom line: Quality business, reasonable price. Add {company} to your watchlist and pull the trigger {_bz}.",
                f"Bottom line: The BUY case for {company} is intact \u2014 initiate or add to your position {_bz}.",
            ])
        elif verdict == 'HOLD':
            if buy_low and cur_price < float(buy_low):
                p4 = pick([
                    f"Bottom line: Hold existing positions. The price is attractive but the overall score ({score_10}/100) doesn't yet justify aggressive buying — wait for fundamentals to improve.",
                    f"Bottom line: Existing holders should stay in. New investors can consider a very small starter position but keep most powder dry until the score improves.",
                    f"Bottom line: {company} is cheap relative to fair value but the score of {score_10}/100 reflects mixed signals — hold what you have and add only if the business shows improvement.",
                ])
            elif buy_low:
                p4 = pick([
                    f"Bottom line: Hold existing positions. New investors should wait for a dip to Rs.{_p(buy_low)}–Rs.{_p(buy_high)} before entering.",
                    f"Bottom line: No need to exit {company} if you own it. If you don't, patience — wait for Rs.{_p(buy_low)} before committing.",
                    f"Bottom line: Current holders are in a good spot. New investors should keep Rs.{_p(buy_low)}–Rs.{_p(buy_high)} as their entry target.",
                ])
            else:
                p4 = pick([
                    f"Bottom line: Hold {company} and reassess after the next quarterly results.",
                    f"Bottom line: Stay put with {company} — no urgent action needed in either direction.",
                    f"Bottom line: Neither a compelling buy nor a reason to sell — hold and monitor.",
                ])
        elif verdict == 'MILD SELL':
            p4 = pick([
                f"Bottom line: Consider trimming your position in {company}. The score of {score_10}/100 suggests risk is building — don't add and review your allocation.",
                f"Bottom line: Not an urgent exit but worth reducing exposure. {company} at {score_10}/100 has better alternatives available in the market.",
                f"Bottom line: Mild caution on {company} — trim rather than exit completely, and set a stop if the score falls further.",
            ])
        else:
            p4 = pick([
                f"Bottom line: Avoid {company} until fundamentals improve or the price corrects significantly.",
                f"Bottom line: The numbers don't support buying {company} right now \u2014 better opportunities exist elsewhere.",
                f"Bottom line: Step aside on {company} for now. Revisit when the score improves above 55/100.",
            ])

        advice_color = C_GREEN if verdict in ('BUY', 'STRONG BUY', 'MILD BUY') else C_RED if verdict in ('SELL', 'MILD SELL') else C_GOLD
        for i, para in enumerate([p1, p2, p3, p4]):
            story.append(Paragraph(para, S(f'adv{i}',
                fontSize=10, textColor=advice_color if i == 0 else C_TEXT, leading=15)))
            story.append(Spacer(1, 3*mm))

    except Exception as _e:
        story.append(Paragraph(
            'Advice unavailable at this time. Please refer to the analysis sections above.',
            S('adv_err', fontSize=10, textColor=C_MUTED)))

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
