import sys
import os
import json
import tempfile
import requests
from math import ceil
from datetime import datetime
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)

import warnings
warnings.filterwarnings('ignore', message='Glyph.*missing from font.*DejaVu Sans')
warnings.filterwarnings('ignore', message='Matplotlib currently does not support Devanagari natively')

plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# FONT SETUP
# ─────────────────────────────────────────────
FONT_DIR = "font"
FONTS = {
    "noto_regular":    os.path.join(FONT_DIR, "NotoSans-Regular.ttf"),
    "noto_bold":       os.path.join(FONT_DIR, "NotoSans-Bold.ttf"),
    "noto_deva":       os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf"),
    "noto_deva_bold":  os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf"),
    "noto_arabic":     os.path.join(FONT_DIR, "NotoSansArabic-Regular.ttf"),
    "noto_arabic_bold":os.path.join(FONT_DIR, "NotoSansArabic-Bold.ttf"),
}

# ─────────────────────────────────────────────
# LAYOUT & COLORS
# ─────────────────────────────────────────────
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN       = 30
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
CHART_W      = 240
CHART_H      = 180

PRIMARY    = HexColor("#1F3A93")
ACCENT     = HexColor("#4A90E2")
LIGHT_BG   = HexColor("#F5F7FA")
TEXT       = HexColor("#222222")
BORDER     = HexColor("#DDDDDD")
BRAND_BLUE = HexColor("#2E86C1")

HIGH_RED    = HexColor("#E74C3C")
MED_ORANGE  = HexColor("#E67E22")
LOW_GREEN   = HexColor("#27AE60")
NEUTRAL_GRAY = HexColor("#7F8C8D")

RL_PALETTE = {
    "positive": HexColor("#4CAF50"),
    "neutral":  HexColor("#FFC107"),
    "negative": HexColor("#F44336"),
    "primary":  HexColor("#1F3A93"),
    "secondary":HexColor("#4A90E2"),
}
MPL_PALETTE = {
    "positive": "#4CAF50",
    "neutral":  "#FFC107",
    "negative": "#F44336",
    "primary":  "#1F3A93",
    "secondary":"#4A90E2",
}

REGISTERED_FONTS = {}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_get_nested(data, *keys, default=None):
    current = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current

def safe_numeric_value(value, default=0):
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
    except (ValueError, TypeError):
        pass
    return default

def safe_list_value(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, (str, int, float)):
        return [value]
    return default

def safe_string_value(value, default=""):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default

def has_content(obj):
    if obj is None:
        return False
    if isinstance(obj, dict):
        return bool(obj) and any(has_content(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return bool(obj) and any(has_content(v) for v in obj)
    if isinstance(obj, str):
        s = obj.strip()
        return s not in ("", "-", "none", "None")
    if isinstance(obj, (int, float, bool)):
        return True
    return True

def validate_data_structure(data):
    if not isinstance(data, dict):
        data = {}
    default_structure = {
        'metadata': {},
        'validation': {},
        'language_info': {},
        'analysis_results': {
            'transcript': {},
            'summary': {},
            'sentiment': {},
            'quality_scores': {},
            'agent_performance': {},
            'issues': {},
            'onboarding': {},
            'topics': {},
            'speakers': {},
            'segment_sentiment': {},
        }
    }
    for key, default_value in default_structure.items():
        if key not in data or data[key] is None:
            data[key] = default_value
    return data


# ─────────────────────────────────────────────
# FONT HELPERS
# ─────────────────────────────────────────────

def register_fonts():
    global REGISTERED_FONTS
    REGISTERED_FONTS = {}
    for alias, path in FONTS.items():
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(alias, path))
                REGISTERED_FONTS[alias] = True
            except Exception:
                REGISTERED_FONTS[alias] = False
        else:
            REGISTERED_FONTS[alias] = False

def choose_font_for_text(text, bold=False):
    if text is None:
        text = ""
    text = str(text)
    has_deva   = any("\u0900" <= ch <= "\u097F" for ch in text)
    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in text)
    if has_arabic:
        return ("noto_arabic_bold" if bold and REGISTERED_FONTS.get("noto_arabic_bold")
                else "noto_arabic"  if REGISTERED_FONTS.get("noto_arabic")
                else "Helvetica-Bold" if bold else "Helvetica")
    elif has_deva:
        return ("noto_deva_bold" if bold and REGISTERED_FONTS.get("noto_deva_bold")
                else "noto_deva"  if REGISTERED_FONTS.get("noto_deva")
                else "Helvetica-Bold" if bold else "Helvetica")
    else:
        return ("noto_bold"    if bold and REGISTERED_FONTS.get("noto_bold")
                else "noto_regular" if REGISTERED_FONTS.get("noto_regular")
                else "Helvetica-Bold" if bold else "Helvetica")

def auto_paragraph(text, styles, base_style='CustomBody', bold=False):
    if text is None:
        text = ""
    text = str(text)
    base = styles[base_style]
    style = base.clone(f"{base_style}_temp")
    style.fontName = choose_font_for_text(text, bold=bold)
    return Paragraph(text, style)


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

def build_styles():
    styles = getSampleStyleSheet()

    def safe_add(style):
        if style.name in styles.byName:
            del styles.byName[style.name]
        styles.add(style)

    safe_add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontName=choose_font_for_text("Hello"),
        fontSize=11, leading=14,
        textColor=HexColor("#2E2E2E"),
        spaceAfter=6, alignment=TA_JUSTIFY
    ))
    safe_add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=22, leading=26,
        textColor=HexColor("#2C3E50"),
        alignment=TA_CENTER, spaceAfter=12
    ))
    safe_add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontName=choose_font_for_text("Hello"),
        fontSize=14, leading=18,
        textColor=PRIMARY, alignment=TA_CENTER
    ))
    safe_add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=16, leading=20, spaceAfter=8
    ))
    safe_add(ParagraphStyle(
        name='SubHeading',
        parent=styles['Heading3'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=12, leading=16,
        textColor=ACCENT, spaceAfter=6
    ))
    return styles


# ─────────────────────────────────────────────
# HEADING BUILDERS
# ─────────────────────────────────────────────

def section_heading(text, styles, bg_color=PRIMARY):
    para = auto_paragraph(f'<font color="white">{text}</font>', styles, base_style='SectionHeading', bold=True)
    table = Table([[para]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), bg_color),
        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME',      (0,0), (-1,-1), 'noto_bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    return table

def executive_heading(text, styles, bg_color=PRIMARY):
    para = auto_paragraph(f'<font color="white">{text}</font>', styles, base_style='SectionHeading', bold=True)
    para.style.alignment = TA_CENTER
    table = Table([[para]], colWidths=[CONTENT_WIDTH])
    table.hAlign = 'CENTER'
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), bg_color),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME',      (0,0), (-1,-1), 'noto_bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 18),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('BOX',           (0,0), (-1,-1), 0.75, HexColor("#154360")),
    ]))
    return table


# ─────────────────────────────────────────────
# LOGO / BRAND
# ─────────────────────────────────────────────

def download_logo_image(logo_url):
    try:
        response = requests.get(logo_url, timeout=10)
        if response.status_code == 200:
            temp_path = os.path.join(tempfile.gettempdir(), "convlyze_logo.png")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            return temp_path
    except Exception as e:
        print(f"Error downloading logo: {e}")
    return None

def create_brand_header():
    logo_url = "https://convlyze-web.s3.ap-south-1.amazonaws.com/uploads/user_images/Oe8h91lUx.png"
    logo_path = download_logo_image(logo_url)
    if logo_path and os.path.exists(logo_path):
        try:
            pil_img = PILImage.open(logo_path)
            orig_w, orig_h = pil_img.size
            desired_h = 60
            desired_w = desired_h * (orig_w / orig_h)
            logo_img = Image(logo_path, width=desired_w, height=desired_h)
            logo_img.hAlign = 'CENTER'
            return logo_img
        except Exception as e:
            print(f"Error creating logo image: {e}")
    fallback_style = ParagraphStyle(
        name='FallbackBrand', fontName='Helvetica-Bold',
        fontSize=24, textColor=BRAND_BLUE, alignment=TA_CENTER
    )
    return Paragraph("Convlyze - AI Feedback Intelligence", fallback_style)


# ─────────────────────────────────────────────
# PRIORITY / SENTIMENT COLOR HELPERS
# ─────────────────────────────────────────────

def get_priority_color(priority):
    p = safe_string_value(priority, "").lower()
    if p == "high":
        return HIGH_RED
    elif p == "medium":
        return MED_ORANGE
    elif p == "low":
        return LOW_GREEN
    return NEUTRAL_GRAY

def get_sentiment_color(sentiment):
    s = safe_string_value(sentiment, "").lower()
    if s == "positive":
        return RL_PALETTE["positive"]
    elif s == "negative":
        return RL_PALETTE["negative"]
    return RL_PALETTE["neutral"]

def get_score_color(score):
    s = safe_numeric_value(score, 0)
    if s >= 75:
        return LOW_GREEN
    elif s >= 50:
        return MED_ORANGE
    return HIGH_RED


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

def safe_pie_chart(dist, outpath, title="Sentiment Distribution"):
    try:
        if not dist or not isinstance(dist, dict):
            dist = {"neutral": 100}
        labels = list(dist.keys())
        values = [safe_numeric_value(dist[k], 0) for k in labels]
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
        if not filtered:
            filtered = [("No Data", 100)]
        labels, values = zip(*filtered)
        colors = [MPL_PALETTE.get(k.lower(), MPL_PALETTE["secondary"]) for k in labels]
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.text(0.5, 0.5, 'Chart unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def safe_bar_chart(scores_dict, outpath, title="Scores"):
    try:
        if not scores_dict or not isinstance(scores_dict, dict):
            scores_dict = {"No Data": 50}
        labels = [k.replace("_score", "").replace("_", " ").title() for k in scores_dict.keys()]
        values = [safe_numeric_value(v, 0) for v in scores_dict.values()]
        colors = [MPL_PALETTE["positive"] if v >= 75 else MPL_PALETTE["neutral"] if v >= 50 else MPL_PALETTE["negative"] for v in values]
        fig, ax = plt.subplots(figsize=(6, max(2.5, len(labels) * 0.5)), dpi=150)
        bars = ax.barh(labels, values, color=colors, alpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (0-100)")
        ax.set_title(title, fontsize=11, fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f'{val:.0f}', va='center', fontsize=9)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        ax.text(0.5, 0.5, 'Chart unavailable', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def safe_timeline_chart(segments, outpath, title="Sentiment Timeline"):
    try:
        if not segments or not isinstance(segments, list):
            return
        times, scores, labels, colors = [], [], [], []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            score = safe_numeric_value(seg.get("sentiment_score"), None)
            sentiment = safe_string_value(seg.get("sentiment", ""), "neutral").lower()
            if score is not None:
                times.append(i)
                scores.append(float(score))
                labels.append(sentiment.title())
                colors.append(MPL_PALETTE.get(sentiment, MPL_PALETTE["secondary"]))
        if not times:
            return
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=150)
        ax.scatter(times, scores, c=colors, s=60, zorder=3)
        ax.plot(times, scores, color=MPL_PALETTE["secondary"], linewidth=1.2, alpha=0.7)
        for x, y, lbl in zip(times, scores, labels):
            ax.text(x, y + 2, lbl, fontsize=7, ha="center")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Sentiment Score")
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating timeline chart: {e}")


# ─────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────

def title_page(data, styles):
    story = []
    data = validate_data_structure(data)

    story.append(Spacer(1, 30))

    # Brand header
    try:
        brand_header = create_brand_header()
        story.append(brand_header)
        story.append(Spacer(1, 40))
    except Exception as e:
        print(f"Error creating brand header: {e}")
        story.append(Paragraph("Convlyze - AI Feedback Intelligence", styles['ReportTitle']))
        story.append(Spacer(1, 30))

    # Title banner
    title_style = styles['ReportTitle'].clone('ReportTitle_banner')
    title_style.textColor = HexColor("#FFFFFF")
    title_style.fontSize = 26
    title_style.leading = 30
    title_para = Paragraph("Feedback Call Analysis Report", title_style)
    title_banner = Table([[title_para]], colWidths=[CONTENT_WIDTH])
    title_banner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), PRIMARY),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 35),
        ('BOTTOMPADDING', (0,0), (-1,-1), 35),
        ('BOX',           (0,0), (-1,-1), 2, HexColor("#154360")),
    ]))
    story.append(title_banner)
    story.append(Spacer(1, 25))

    # Metadata table
    meta = safe_get_nested(data, 'metadata', default={})

    def format_duration(seconds):
        try:
            s = safe_numeric_value(seconds, 0)
            if s < 60:
                return f"{s:.1f} seconds"
            s_int = int(round(s))
            h, m, sec = s_int // 3600, (s_int % 3600) // 60, s_int % 60
            if h:  return f"{h}h {m}m {sec}s"
            if m:  return f"{m}m {sec}s"
            return f"{sec}s"
        except Exception:
            return str(seconds) if seconds else "-"

    def format_timestamp(ts_str):
        try:
            if ts_str and ts_str != "-":
                if "T" in ts_str:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    return dt.strftime("%B %d, %Y at %I:%M %p")
                return ts_str
            return "-"
        except Exception:
            return str(ts_str) if ts_str else "-"

    file_url  = safe_string_value(meta.get("file_url", ""))
    file_name = safe_string_value(meta.get("file_name", "")) or (os.path.basename(urlparse(file_url).path) if file_url else "Unknown File")
    ts                = format_timestamp(safe_string_value(meta.get("timestamp", "-")))
    proc_time         = format_duration(meta.get("processing_time_seconds", 0))
    primary_language  = safe_string_value(meta.get("primary_language", "Unknown")).title()
    analysis_version  = safe_string_value(meta.get("analysis_version", "1.0"))
    analysis_type     = safe_string_value(meta.get("analysis_type", "feedback_call"))

    summary_data  = safe_get_nested(data, 'analysis_results', 'summary', default={})
    agent_name    = safe_string_value(summary_data.get("agent_name", ""), "Not Identified")
    employee_name = safe_string_value(summary_data.get("employee_name", ""), "Not Identified")
    organization  = safe_string_value(summary_data.get("organization", ""), "Not Specified")
    call_outcome  = safe_string_value(summary_data.get("call_outcome", ""), "Unknown").replace("_", " ").title()

    meta_data = [
        [auto_paragraph("📄 File Name",        styles, bold=True), auto_paragraph(file_name, styles)],
        [auto_paragraph("📅 Generated On",     styles, bold=True), auto_paragraph(ts, styles)],
        [auto_paragraph("🏢 Organization",     styles, bold=True), auto_paragraph(organization, styles)],
        [auto_paragraph("👤 HR Agent",         styles, bold=True), auto_paragraph(agent_name, styles)],
        [auto_paragraph("🙋 Employee",         styles, bold=True), auto_paragraph(employee_name, styles)],
        [auto_paragraph("📋 Call Outcome",     styles, bold=True), auto_paragraph(call_outcome, styles)],
        [auto_paragraph("⏱️ Processing Time",  styles, bold=True), auto_paragraph(proc_time, styles)],
        [auto_paragraph("🌐 Primary Language", styles, bold=True), auto_paragraph(primary_language, styles)],
        [auto_paragraph("🔧 Analysis Version", styles, bold=True), auto_paragraph(analysis_version, styles)],
        [auto_paragraph("📋 Analysis Type",    styles, bold=True), auto_paragraph(analysis_type.replace("_", " ").title(), styles)],
    ]

    meta_table = Table(meta_data, colWidths=[180, CONTENT_WIDTH - 220])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), HexColor("#FFFFFF")),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ('BOX',           (0,0), (-1,-1), 1.5, BORDER),
        ('INNERGRID',     (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
        ('TEXTCOLOR',     (0,0), (-1,-1), TEXT),
        ('FONTNAME',      (0,0), (-1,-1), 'noto_regular'),
        ('FONTSIZE',      (0,0), (-1,-1), 11),
        ('LEFTPADDING',   (0,0), (-1,-1), 15),
        ('RIGHTPADDING',  (0,0), (-1,-1), 15),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 40))
    footer_style = ParagraphStyle(
        name='FooterNote', fontName='Helvetica-Oblique', fontSize=9,
        textColor=HexColor("#7F8C8D"), alignment=TA_CENTER
    )
    story.append(Paragraph("This report was automatically generated using advanced AI analysis for HR feedback calls", footer_style))
    story.append(Spacer(1, 30))
    return story


# ─────────────────────────────────────────────
# MAIN PDF BUILDER
# ─────────────────────────────────────────────

def create_feedback_pdf_from_json(data, out_pdf_path):
    """Create feedback call PDF report from JSON data"""
    try:
        if data is None:
            data = {}
        data = validate_data_structure(data)

        register_fonts()
        styles = build_styles()

        doc = SimpleDocTemplate(
            out_pdf_path, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN
        )
        story = []

        # ── Title Page ──────────────────────────────────────
        try:
            story.extend(title_page(data, styles))
            story.append(PageBreak())
        except Exception as e:
            print(f"Error creating title page: {e}")
            story.append(Paragraph("Feedback Call Analysis Report", styles['ReportTitle']))
            story.append(Spacer(1, 20))
            story.append(PageBreak())

        analysis = safe_get_nested(data, 'analysis_results', default={})

        if has_content(analysis):
            story.append(executive_heading("Feedback Call Analysis Results", styles))
            story.append(Spacer(1, 20))

        # ── 1. EXECUTIVE SUMMARY ────────────────────────────
        summary_data = safe_get_nested(analysis, 'summary', default={})
        if has_content(summary_data):
            story.append(section_heading("Executive Summary", styles, bg_color=PRIMARY))
            story.append(Spacer(1, 12))

            exec_summary = safe_string_value(summary_data.get("executive_summary", ""))
            if exec_summary:
                story.append(auto_paragraph(exec_summary, styles))
                story.append(Spacer(1, 10))

            call_purpose = safe_string_value(summary_data.get("call_purpose", ""))
            if call_purpose:
                story.append(auto_paragraph(f"<b>Call Purpose:</b> {call_purpose}", styles))
                story.append(Spacer(1, 6))

            experience_rating = safe_string_value(summary_data.get("overall_experience_rating", ""))
            if experience_rating:
                story.append(auto_paragraph(f"<b>Overall Experience Rating:</b> {experience_rating.title()}", styles))
                story.append(Spacer(1, 6))

            key_takeaways = safe_list_value(summary_data.get("key_takeaways", []))
            if key_takeaways:
                story.append(auto_paragraph("Key Takeaways", styles, base_style='SubHeading'))
                for takeaway in key_takeaways:
                    story.append(auto_paragraph(f"• {safe_string_value(takeaway)}", styles))
                story.append(Spacer(1, 12))

        # ── 2. ONBOARDING EXPERIENCE ────────────────────────
        onboarding_data = safe_get_nested(analysis, 'onboarding', default={})
        if has_content(onboarding_data):
            story.append(section_heading("Onboarding Experience", styles))
            story.append(Spacer(1, 12))

            ob_scores = onboarding_data.get("onboarding_scores", {})
            if has_content(ob_scores):
                overall_ob = safe_numeric_value(ob_scores.get("overall_onboarding_score"), None)
                if overall_ob is not None:
                    oc = get_score_color(overall_ob)
                    ob_banner = auto_paragraph(
                        f'<font color="white"><b>Overall Onboarding Score: {overall_ob:.0f}%</b></font>',
                        styles, base_style='CustomBody'
                    )
                    ob_banner.style.alignment = TA_CENTER
                    ob_table = Table([[ob_banner]], colWidths=[CONTENT_WIDTH])
                    ob_table.setStyle(TableStyle([
                        ('BACKGROUND',    (0,0), (-1,-1), oc),
                        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING',    (0,0), (-1,-1), 12),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                        ('BOX',           (0,0), (-1,-1), 1, BORDER),
                    ]))
                    story.append(ob_table)
                    story.append(Spacer(1, 12))

                dim_scores = {k: v for k, v in ob_scores.items() if k != "overall_onboarding_score" and v is not None}
                if dim_scores:
                    try:
                        ob_bar_path = os.path.join(tempfile.gettempdir(), "feedback_onboarding_bar.png")
                        safe_bar_chart(dim_scores, ob_bar_path, title="Onboarding Dimension Scores")
                        story.append(Image(ob_bar_path, width=360, height=180))
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        print(f"Onboarding bar chart error: {e}")

            highlights = onboarding_data.get("experience_highlights", {})
            if has_content(highlights):
                positives   = safe_list_value(highlights.get("positives", []))
                concerns    = safe_list_value(highlights.get("concerns", []))
                suggestions = safe_list_value(highlights.get("suggestions", []))

                if positives:
                    story.append(auto_paragraph("Positives", styles, base_style='SubHeading'))
                    for p in positives:
                        story.append(auto_paragraph(f"✔ {safe_string_value(p)}", styles))
                    story.append(Spacer(1, 8))

                if concerns:
                    story.append(auto_paragraph("Concerns", styles, base_style='SubHeading'))
                    for c in concerns:
                        story.append(auto_paragraph(f"▸ {safe_string_value(c)}", styles))
                    story.append(Spacer(1, 8))

                if suggestions:
                    story.append(auto_paragraph("Suggestions", styles, base_style='SubHeading'))
                    for s in suggestions:
                        story.append(auto_paragraph(f"💡 {safe_string_value(s)}", styles))
                    story.append(Spacer(1, 8))

            engagement = onboarding_data.get("employee_engagement", {})
            if has_content(engagement):
                eng_level  = safe_string_value(engagement.get("engagement_level", ""))
                first_imp  = safe_string_value(engagement.get("first_impression", ""))
                likelihood = safe_string_value(engagement.get("likelihood_to_stay", ""))
                if eng_level:
                    story.append(auto_paragraph(f"<b>Engagement Level:</b> {eng_level.title()}", styles))
                if likelihood:
                    story.append(auto_paragraph(f"<b>Likelihood to Stay:</b> {likelihood.title()}", styles))
                if first_imp:
                    story.append(auto_paragraph(f"<b>First Impression:</b> {first_imp}", styles))
                story.append(Spacer(1, 12))

        # ── 3. SENTIMENT & QUALITY OVERVIEW ─────────────────
        sentiment_data = safe_get_nested(analysis, 'sentiment', default={})
        quality_data   = safe_get_nested(analysis, 'quality_scores', default={})

        if has_content(sentiment_data) or has_content(quality_data):
            story.append(section_heading("Sentiment & Quality Overview", styles))
            story.append(Spacer(1, 12))

        if has_content(sentiment_data):
            overall_sentiment = safe_string_value(sentiment_data.get("overall_sentiment", ""))
            sentiment_score   = safe_numeric_value(sentiment_data.get("sentiment_score"), None)

            if overall_sentiment or sentiment_score is not None:
                sent_color  = get_sentiment_color(overall_sentiment)
                sent_label  = overall_sentiment.title() if overall_sentiment else "N/A"
                score_label = f"{sentiment_score:.0f}%" if sentiment_score is not None else "N/A"

                sent_para = auto_paragraph(
                    f'<font color="white"><b>Employee Sentiment: {sent_label} — {score_label}</b></font>',
                    styles, base_style='CustomBody'
                )
                sent_para.style.alignment = TA_CENTER
                sent_banner = Table([[sent_para]], colWidths=[CONTENT_WIDTH])
                sent_banner.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0), (-1,-1), sent_color),
                    ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
                    ('TOPPADDING',    (0,0), (-1,-1), 14),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 14),
                    ('BOX',           (0,0), (-1,-1), 1, BORDER),
                ]))
                story.append(sent_banner)
                story.append(Spacer(1, 12))

            emotional_tones = safe_list_value(sentiment_data.get("emotional_tones", []))
            if emotional_tones:
                story.append(auto_paragraph(f"<b>Emotional Tones:</b> {', '.join(emotional_tones)}", styles))
                story.append(Spacer(1, 8))

            dist = sentiment_data.get("sentiment_distribution", {})
            if has_content(dist):
                try:
                    pie_path = os.path.join(tempfile.gettempdir(), "feedback_sentiment_pie.png")
                    safe_pie_chart(dist, pie_path, title="Sentiment Distribution")
                    story.append(Image(pie_path, width=220, height=165))
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Pie chart error: {e}")

        if has_content(quality_data):
            story.append(auto_paragraph("Quality Scores", styles, base_style='SubHeading'))

            scores_map = {
                "audio_quality_score":      quality_data.get("audio_quality_score"),
                "speech_clarity_score":     quality_data.get("speech_clarity_score"),
                "transcription_confidence": quality_data.get("transcription_confidence"),
                "analysis_reliability":     quality_data.get("analysis_reliability"),
            }
            scores_map = {k: v for k, v in scores_map.items() if v is not None}

            if scores_map:
                try:
                    bar_path = os.path.join(tempfile.gettempdir(), "feedback_quality_bar.png")
                    safe_bar_chart(scores_map, bar_path, title="Quality Scores")
                    story.append(Image(bar_path, width=360, height=180))
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Quality bar chart error: {e}")

            noise_level = safe_string_value(quality_data.get("noise_level", ""))
            if noise_level:
                story.append(auto_paragraph(f"<b>Noise Level:</b> {noise_level.title()}", styles))

            overall_score = safe_numeric_value(quality_data.get("overall_score"), None)
            if overall_score is not None:
                story.append(auto_paragraph(f"<b>Overall Quality Score:</b> {overall_score:.0f}%", styles))

            speaker_confs = safe_list_value(quality_data.get("confidence_scores_by_speaker", []))
            if speaker_confs:
                story.append(Spacer(1, 8))
                story.append(auto_paragraph("<b>Speaker Confidence:</b>", styles))
                for sc in speaker_confs:
                    if isinstance(sc, dict):
                        spk   = safe_string_value(sc.get("speaker", ""), "Unknown")
                        score = safe_numeric_value(sc.get("score"), 0)
                        story.append(auto_paragraph(f"  • {spk}: {score:.0f}%", styles))
            story.append(Spacer(1, 16))

        # ── 4. ISSUES & ACTION ITEMS ────────────────────────
        issues_data = safe_get_nested(analysis, 'issues', default={})
        if has_content(issues_data):
            story.append(section_heading("Issues & Action Items", styles))
            story.append(Spacer(1, 12))

            issues_list = safe_list_value(issues_data.get("issues", []))
            total_issues = safe_numeric_value(issues_data.get("total_issues_count"), len(issues_list))
            high_priority = safe_numeric_value(issues_data.get("high_priority_count"), 0)

            if total_issues > 0:
                story.append(auto_paragraph(
                    f"<b>Total Issues Raised:</b> {int(total_issues)}  |  <b>High Priority:</b> {int(high_priority)}",
                    styles
                ))
                story.append(Spacer(1, 8))

            if issues_list:
                issue_rows = [[
                    auto_paragraph("Issue", styles, bold=True),
                    auto_paragraph("Raised By", styles, bold=True),
                    auto_paragraph("Category", styles, bold=True),
                    auto_paragraph("Priority", styles, bold=True),
                ]]
                for issue in issues_list:
                    if isinstance(issue, dict):
                        title_text = safe_string_value(issue.get("issue_title", ""), "-")
                        desc       = safe_string_value(issue.get("description", ""), "")
                        combined   = f"<b>{title_text}</b><br/>{desc}" if desc else title_text
                        raised_by  = safe_string_value(issue.get("raised_by", ""), "-")
                        category   = safe_string_value(issue.get("category", ""), "-").title()
                        priority   = safe_string_value(issue.get("priority", ""), "-").upper()
                        issue_rows.append([
                            auto_paragraph(combined, styles),
                            auto_paragraph(raised_by, styles),
                            auto_paragraph(category, styles),
                            auto_paragraph(priority, styles),
                        ])

                if len(issue_rows) > 1:
                    issue_table = Table(issue_rows, colWidths=[CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.18, CONTENT_WIDTH*0.20, CONTENT_WIDTH*0.17])
                    issue_table.setStyle(TableStyle([
                        ('BACKGROUND',    (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR',     (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME',      (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE',      (0,0), (-1,-1), 9),
                        ('BOX',           (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID',     (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING',   (0,0), (-1,-1), 6),
                        ('TOPPADDING',    (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(issue_table)
                    story.append(Spacer(1, 14))

            # Action items
            action_items = safe_list_value(issues_data.get("action_items", []))
            if action_items:
                story.append(auto_paragraph("Action Items", styles, base_style='SubHeading'))
                action_rows = [[
                    auto_paragraph("Task", styles, bold=True),
                    auto_paragraph("Assignee", styles, bold=True),
                    auto_paragraph("Deadline", styles, bold=True),
                    auto_paragraph("Priority", styles, bold=True),
                ]]
                for item in action_items:
                    if isinstance(item, dict):
                        action_rows.append([
                            auto_paragraph(safe_string_value(item.get("task", ""), "-"), styles),
                            auto_paragraph(safe_string_value(item.get("assignee", ""), "-"), styles),
                            auto_paragraph(safe_string_value(item.get("deadline", ""), "-").replace("_", " ").title(), styles),
                            auto_paragraph(safe_string_value(item.get("priority", ""), "-").upper(), styles),
                        ])
                if len(action_rows) > 1:
                    action_table = Table(action_rows, colWidths=[CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.2, CONTENT_WIDTH*0.2, CONTENT_WIDTH*0.15])
                    action_table.setStyle(TableStyle([
                        ('BACKGROUND',    (0,0), (-1,0), PRIMARY),
                        ('TEXTCOLOR',     (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME',      (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE',      (0,0), (-1,-1), 9),
                        ('BOX',           (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID',     (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING',   (0,0), (-1,-1), 6),
                        ('TOPPADDING',    (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(action_table)
                    story.append(Spacer(1, 14))

            # Call quality issues
            cq = issues_data.get("call_quality_issues", {})
            if has_content(cq):
                cq_notes = safe_string_value(cq.get("notes", ""))
                dropped  = cq.get("call_dropped", False)
                incomplete = cq.get("incomplete_conversation", False)
                if dropped or incomplete or cq_notes:
                    story.append(auto_paragraph("Call Quality Issues", styles, base_style='SubHeading'))
                    if dropped:
                        story.append(auto_paragraph("⚠ Call was dropped during the conversation", styles))
                    if incomplete:
                        story.append(auto_paragraph("⚠ Conversation was incomplete", styles))
                    if cq_notes:
                        story.append(auto_paragraph(f"Notes: {cq_notes}", styles))
                    story.append(Spacer(1, 12))

        # ── 5. KEY TOPICS DISCUSSED ─────────────────────────
        topics_data = safe_get_nested(analysis, 'topics', default={})
        if has_content(topics_data):
            key_topics = safe_list_value(topics_data.get("key_topics", []))
            if key_topics:
                story.append(section_heading("Key Topics Discussed", styles))
                story.append(Spacer(1, 12))

                most_discussed = safe_string_value(topics_data.get("most_discussed_topic", ""))
                if most_discussed:
                    story.append(auto_paragraph(f"<b>Most Discussed:</b> {most_discussed}", styles))
                    story.append(Spacer(1, 8))

                topic_rows = [[
                    auto_paragraph("Topic", styles, bold=True),
                    auto_paragraph("Importance", styles, bold=True),
                    auto_paragraph("Mentions", styles, bold=True),
                    auto_paragraph("Sentiment", styles, bold=True),
                ]]
                for topic in key_topics:
                    if isinstance(topic, dict):
                        topic_rows.append([
                            auto_paragraph(safe_string_value(topic.get("topic", ""), "-"), styles),
                            auto_paragraph(f"{safe_numeric_value(topic.get('importance_score'), 0):.0f}%", styles),
                            auto_paragraph(str(int(safe_numeric_value(topic.get("mentions"), 0))), styles),
                            auto_paragraph(safe_string_value(topic.get("sentiment", ""), "-").title(), styles),
                        ])

                if len(topic_rows) > 1:
                    topic_table = Table(topic_rows, colWidths=[CONTENT_WIDTH*0.40, CONTENT_WIDTH*0.20, CONTENT_WIDTH*0.20, CONTENT_WIDTH*0.20])
                    topic_table.setStyle(TableStyle([
                        ('BACKGROUND',    (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR',     (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME',      (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE',      (0,0), (-1,-1), 9),
                        ('BOX',           (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID',     (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING',   (0,0), (-1,-1), 6),
                        ('TOPPADDING',    (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(topic_table)
                    story.append(Spacer(1, 14))

        # ── 6. TRANSCRIPT ───────────────────────────────────
        transcript_data = safe_get_nested(analysis, 'transcript', default={})
        # Always render the transcript section — fall back to full text if segments are missing
        story.append(section_heading("Call Transcript", styles))
        story.append(Spacer(1, 10))

        segments = safe_list_value(transcript_data.get("segments", []) if isinstance(transcript_data, dict) else [])
        full_text = safe_string_value(transcript_data.get("transcript", "") if isinstance(transcript_data, dict) else "", "")

        if segments:
            trans_rows = [[
                auto_paragraph("Speaker", styles, bold=True),
                auto_paragraph("Time", styles, bold=True),
                auto_paragraph("Text", styles, bold=True),
            ]]
            for seg in segments:
                if isinstance(seg, dict):
                    speaker = safe_string_value(seg.get("speaker_name", ""), "-")
                    time    = safe_string_value(seg.get("start_time", ""), "-")
                    text    = safe_string_value(seg.get("text", ""), "-")
                    trans_rows.append([
                        auto_paragraph(speaker, styles),
                        auto_paragraph(time, styles),
                        auto_paragraph(text, styles),
                    ])

            if len(trans_rows) > 1:
                trans_table = Table(trans_rows, colWidths=[80, 55, CONTENT_WIDTH - 155])
                trans_table.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0), (-1,0), ACCENT),
                    ('TEXTCOLOR',     (0,0), (-1,0), HexColor("#FFFFFF")),
                    ('FONTNAME',      (0,0), (-1,0), 'noto_bold'),
                    ('FONTSIZE',      (0,0), (-1,0), 11),
                    ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                    ('FONTSIZE',      (0,1), (-1,-1), 9),
                    ('FONTNAME',      (0,1), (-1,-1), 'noto_regular'),
                    ('ROWBACKGROUNDS',(0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                    ('INNERGRID',     (0,0), (-1,-1), 0.25, BORDER),
                    ('BOX',           (0,0), (-1,-1), 0.5, BORDER),
                    ('LEFTPADDING',   (0,0), (-1,-1), 6),
                    ('RIGHTPADDING',  (0,0), (-1,-1), 6),
                    ('TOPPADDING',    (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ]))
                story.append(trans_table)
                story.append(Spacer(1, 18))
        elif full_text:
            # Fallback: render full transcript as plain text
            story.append(auto_paragraph(full_text, styles))
            story.append(Spacer(1, 18))
        else:
            story.append(auto_paragraph("Transcript not available for this call.", styles))
            story.append(Spacer(1, 18))

        # ── Build PDF ────────────────────────────────────────
        try:
            doc.build(story)
            return out_pdf_path
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise

    except Exception as e:
        print(f"Critical error in feedback PDF creation: {e}")
        try:
            doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
            story = [
                Paragraph("Feedback Call Analysis Report", getSampleStyleSheet()['Title']),
                Spacer(1, 20),
                Paragraph("Error occurred while generating full report.", getSampleStyleSheet()['Normal']),
                Paragraph(f"Error details: {str(e)}", getSampleStyleSheet()['Normal'])
            ]
            doc.build(story)
            return out_pdf_path
        except Exception as fallback_error:
            print(f"Even fallback PDF creation failed: {fallback_error}")
            raise


# ─────────────────────────────────────────────
# UPLOAD FUNCTION (mirrors report1.py)
# ─────────────────────────────────────────────

def generate_and_upload_feedback_report(data):
    """Generate feedback call PDF report and upload to API"""
    api_url = "https://api.convlyze.com/api/users/upload"

    try:
        if data is None:
            print("Warning: No data provided for feedback report generation")
            data = {}

        if not isinstance(data, dict):
            print(f"Warning: Input data is not a dict (type: {type(data)}), converting...")
            data = {}

        data = validate_data_structure(data)
        out_pdf = os.path.join(tempfile.gettempdir(), "convlyze_feedback_call_report.pdf")
        create_feedback_pdf_from_json(data, out_pdf)

        if not os.path.exists(out_pdf):
            raise Exception("PDF file was not created successfully")

        with open(out_pdf, "rb") as f:
            try:
                response = requests.post(api_url, files={"file": f}, timeout=30)
            except requests.exceptions.RequestException as e:
                raise Exception(f"Upload failed: {e}")

        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code}, {response.text}")

        try:
            result = response.json()
            url = result.get("url") or result.get("file_url")
            if not url:
                raise Exception("No URL returned from API")
            return url
        except Exception as e:
            raise Exception(f"Invalid response format: {response.text}") from e

    except Exception as e:
        print(f"Error in generate_and_upload_feedback_report: {e}")
        raise


__all__ = [
    'create_feedback_pdf_from_json',
    'generate_and_upload_feedback_report',
    'validate_data_structure',
    'safe_get_nested',
    'safe_numeric_value',
    'safe_list_value',
    'safe_string_value',
    'has_content',
]