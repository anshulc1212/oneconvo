import sys
import os
import json
import tempfile
import requests
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from math import ceil
from zoneinfo import ZoneInfo
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from urllib.parse import urlparse
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


FONT_DIR = "font"
FONTS = {
    "noto_regular": os.path.join(FONT_DIR, "NotoSans-Regular.ttf"),
    "noto_bold": os.path.join(FONT_DIR, "NotoSans-Bold.ttf"),
    "noto_deva": os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf"),
    "noto_deva_bold": os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf"),
    "noto_arabic": os.path.join(FONT_DIR, "NotoSansArabic-Regular.ttf"),
    "noto_arabic_bold": os.path.join(FONT_DIR, "NotoSansArabic-Bold.ttf"),
}


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 30
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
CHART_W = 240
CHART_H = 180


PRIMARY = HexColor("#1F3A93")
ACCENT = HexColor("#4A90E2")
CARD_BG = HexColor("#FFFFFF")
LIGHT_BG = HexColor("#F5F7FA")
TEXT = HexColor("#222222")
BORDER = HexColor("#DDDDDD")
BRAND_BLUE = HexColor("#2E86C1")
HIRE_GREEN = HexColor("#27AE60")
MAYBE_YELLOW = HexColor("#F39C12")
NO_HIRE_RED = HexColor("#E74C3C")

RL_PALETTE = {
    "neutral_bg": HexColor("#F8F9FA"),
    "border": HexColor("#DDDDDD"),
    "text": HexColor("#222222"),
    "primary": HexColor("#1F3A93"),
    "secondary": HexColor("#4A90E2"),
    "positive": HexColor("#4CAF50"),
    "neutral": HexColor("#FFC107"),
    "negative": HexColor("#F44336"),
    "hire": HexColor("#27AE60"),
    "maybe": HexColor("#F39C12"),
    "no_hire": HexColor("#E74C3C"),
}

MPL_PALETTE = {
    "neutral_bg": "#F8F9FA",
    "border": "#DDDDDD",
    "text": "#222222",
    "primary": "#1F3A93",
    "secondary": "#4A90E2",
    "positive": "#4CAF50",
    "neutral": "#FFC107",
    "negative": "#F44336",
    "hire": "#27AE60",
    "maybe": "#F39C12",
    "no_hire": "#E74C3C",
}

REGISTERED_FONTS = {}


def validate_data_structure(data):
    """Validate and sanitize input data structure"""
    if data is None:
        print("Warning: Input data is None, using default structure")
        data = {}
    
    if not isinstance(data, dict):
        print(f"Warning: Input data is not a dict (type: {type(data)}), converting to dict")
        data = {}
    
    default_structure = {
        'validation': {},
        'metadata': {},
        'language_info': {},
        'analysis_results': {
            'transcript': {},
            'summary': {},
            'sentiment': {},
            'topics': {},
            'speakers': {},
            'segment_sentiment': {},
            'quality_scores': {},
            'action_items': {},
            'emotions': {},
            'candidate_assessment': {},
            'interviewer_quality': {},
            'hiring_recommendation': {}
        }
    }
    
    for key, default_value in default_structure.items():
        if key not in data:
            data[key] = default_value
        elif data[key] is None:
            data[key] = default_value
        elif isinstance(default_value, dict) and not isinstance(data[key], dict):
            data[key] = default_value
    
    return data

def safe_get_nested(data, *keys, default=None):
    """Safely get nested dictionary values"""
    current = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current

def safe_numeric_value(value, default=0):
    """Safely convert value to numeric"""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        if isinstance(value, dict):
            return default
    except (ValueError, TypeError, AttributeError):
        pass
    return default

def safe_list_value(value, default=None):
    """Safely get list value"""
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
    """Safely get string value"""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default

def has_content(obj):
    """Check if object has meaningful content"""
    if obj is None:
        return False
    if isinstance(obj, dict):
        if not obj:
            return False
        return any(has_content(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        if not obj:
            return False
        return any(has_content(v) for v in obj)
    if isinstance(obj, str):
        s = obj.strip()
        return s != "" and s != "-" and s.lower() != "none"
    if isinstance(obj, (int, float, bool)):
        return True
    return True

def register_fonts():
    """Register custom fonts"""
    global REGISTERED_FONTS
    REGISTERED_FONTS = {}
    
    for alias, path in FONTS.items():
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(alias, path))
                REGISTERED_FONTS[alias] = True
            except Exception as e:
                REGISTERED_FONTS[alias] = False
        else:
            REGISTERED_FONTS[alias] = False

def choose_font_for_text(text, bold=False):
    """Choose appropriate font for text"""
    if text is None:
        text = ""
    text = str(text)
    
    has_deva = any("\u0900" <= ch <= "\u097F" for ch in text)
    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in text)
    
    if has_arabic:
        return (
            "noto_arabic_bold" if bold and REGISTERED_FONTS.get("noto_arabic_bold")
            else "noto_arabic" if REGISTERED_FONTS.get("noto_arabic")
            else "Helvetica-Bold" if bold else "Helvetica"
        )
    elif has_deva:
        return (
            "noto_deva_bold" if bold and REGISTERED_FONTS.get("noto_deva_bold")
            else "noto_deva" if REGISTERED_FONTS.get("noto_deva")
            else "Helvetica-Bold" if bold else "Helvetica"
        )
    else:
        return (
            "noto_bold" if bold and REGISTERED_FONTS.get("noto_bold")
            else "noto_regular" if REGISTERED_FONTS.get("noto_regular")
            else "Helvetica-Bold" if bold else "Helvetica"
        )

def auto_paragraph(text, styles, base_style='CustomBody', bold=False):
    """Create paragraph with safe text handling"""
    if text is None:
        text = ""
    text = str(text)
    
    base = styles[base_style]
    style = base.clone(f"{base_style}_temp")
    style.fontName = choose_font_for_text(text, bold=bold)
    return Paragraph(text, style)

def section_heading(text, styles, bg_color=PRIMARY):
    """Create section heading"""
    para = auto_paragraph(
        f'<font color="white">{text}</font>',
        styles,
        base_style='SectionHeading',
        bold=True
    )
    
    table = Table([[para]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,-1), HexColor("#FFFFFF")),
        ('FONTNAME', (0,0), (-1,-1), 'noto_bold'),
        ('FONTSIZE', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    return table

def executive_heading(text, styles, bg_color=PRIMARY):
    """Create executive heading"""
    para = auto_paragraph(
        f'<font color="white">{text}</font>',
        styles,
        base_style='SectionHeading',
        bold=True
    )
    para.style.alignment = TA_CENTER
    
    table = Table([[para]], colWidths=[CONTENT_WIDTH])
    table.hAlign = 'CENTER'
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,-1), HexColor("#FFFFFF")),
        ('FONTNAME', (0,0), (-1,-1), 'noto_bold'),
        ('FONTSIZE', (0,0), (-1,-1), 18),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 0.75, HexColor("#154360")),
    ]))
    return table

def build_styles():
    """Build custom paragraph styles"""
    styles = getSampleStyleSheet()
    
    def safe_add(style):
        if style.name in styles.byName:
            del styles.byName[style.name]
        styles.add(style)
    
    safe_add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontName=choose_font_for_text("Hello"),
        fontSize=11,
        leading=14,
        textColor=HexColor("#2E2E2E"),
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))
    
    safe_add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=22,
        leading=26,
        textColor=HexColor("#2C3E50"),
        alignment=TA_CENTER,
        spaceAfter=12
    ))
    
    safe_add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontName=choose_font_for_text("Hello"),
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        alignment=TA_CENTER
    ))
    
    safe_add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=16,
        leading=20,
        spaceAfter=8
    ))
    
    safe_add(ParagraphStyle(
        name='SubHeading',
        parent=styles['Heading3'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=12,
        leading=16,
        textColor=ACCENT,
        spaceAfter=6
    ))
    
    return styles

def get_recommendation_color(recommendation):
    """Get color based on recommendation"""
    rec = safe_string_value(recommendation, "").upper()
    if "HIRE" in rec and "NOT" not in rec and "NO" not in rec:
        return HIRE_GREEN
    elif "MAYBE" in rec or "DISCUSS" in rec:
        return MAYBE_YELLOW
    elif "NOT" in rec or "REJECT" in rec or "NO" in rec:
        return NO_HIRE_RED
    return ACCENT

def download_logo_image(logo_url):
    """Download and save logo image, returns local path or None on failure"""
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
    """Create Convlyze brand header with logo, falling back to text if unavailable"""
    logo_url = "https://onereap.s3.ap-south-1.amazonaws.com/assets/onetab.aiLogo.png"

    # FIX: download the logo first, then assign logo_path
    logo_path = download_logo_image(logo_url)

    if logo_path and os.path.exists(logo_path):
        try:
            pil_img = PILImage.open(logo_path)
            orig_width, orig_height = pil_img.size

            desired_height = 60
            aspect_ratio = orig_width / orig_height
            desired_width = desired_height * aspect_ratio

            logo_img = Image(logo_path, width=desired_width, height=desired_height)
            logo_img.hAlign = 'CENTER'
            return logo_img
        except Exception as e:
            print(f"Error creating logo image: {e}")

    # Fallback to text if logo download/render failed
    fallback_style = ParagraphStyle(
        name='FallbackBrand',
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=BRAND_BLUE,
        alignment=TA_CENTER
    )
    return Paragraph("OneConvo - AI Interview Intelligence", fallback_style)

def safe_radar_chart(categories_scores, outpath, title="Assessment Radar"):
    """Create radar chart for candidate assessment"""
    try:
        if not categories_scores or not isinstance(categories_scores, dict):
            categories_scores = {"Communication": 50, "Technical": 50, "Cultural Fit": 50}
        
        labels = list(categories_scores.keys())
        values = [safe_numeric_value(v, 50) for v in categories_scores.values()]
        
        if not labels:
            labels, values = ["No Data"], [50]
        
        N = len(labels)
        values += values[:1]
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]
        
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=MPL_PALETTE["primary"])
        ax.fill(angles, values, alpha=0.25, color=MPL_PALETTE["primary"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(title, y=1.08, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Chart data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def title_page(data, styles):
    """Create title page"""
    story = []
    data = validate_data_structure(data)
    
    story.append(Spacer(1, 30))

    try:
        brand_header = create_brand_header()
        story.append(brand_header)
        story.append(Spacer(1, 40))
    except Exception as e:
        print(f"Error creating brand header: {e}")
        fallback_style = styles['ReportTitle'].clone('FallbackBrand')
        fallback_style.textColor = BRAND_BLUE
        fallback_style.fontSize = 24
        story.append(Paragraph("Convlyze - AI Interview Intelligence", fallback_style))
        story.append(Spacer(1, 30))
    

    title_style = styles['ReportTitle'].clone('ReportTitle_banner')
    title_style.textColor = HexColor("#FFFFFF")
    title_style.fontSize = 26
    title_style.leading = 30
    
    title_para = Paragraph(" Interview Analysis", title_style)
    
    title_banner = Table([[title_para]], colWidths=[CONTENT_WIDTH])
    title_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 35),
        ('BOTTOMPADDING', (0,0), (-1,-1), 35),
        ('BOX', (0,0), (-1,-1), 2, HexColor("#154360")),
    ]))
    story.append(title_banner)
    story.append(Spacer(1, 25))
    
    meta = safe_get_nested(data, 'metadata', default={})
    
    def format_duration(seconds):
        try:
            s = safe_numeric_value(seconds, 0)
            if s < 60:
                return f"{s:.1f} seconds"
            s_int = int(round(s))
            h, m, sec = s_int // 3600, (s_int % 3600) // 60, s_int % 60
            if h: return f"{h}h {m}m {sec}s"
            if m: return f"{m}m {sec}s"
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
    
    proc_time = format_duration(meta.get("processing_time_seconds", 0))
    file_url = safe_string_value(meta.get("file_url", ""))
    file_name = os.path.basename(urlparse(file_url).path) if file_url else "Unknown File"
    ts = format_timestamp(safe_string_value(meta.get("timestamp", "-")))
    primary_language = safe_string_value(meta.get("primary_language", "Unknown")).title()
    analysis_version = safe_string_value(meta.get("analysis_version", "v2.0"))
    analysis_type = safe_string_value(meta.get("analysis_type", "recruitment_interview"))

    summary_data = safe_get_nested(data, 'analysis_results', 'summary', default={})
    applied_role = safe_string_value(summary_data.get("applied_role", ""), "Not Specified")
    
    meta_header = Paragraph("Report Details", styles['SubHeading'])
    story.append(meta_header)
    story.append(Spacer(1, 10))

    meta_data = [
        [auto_paragraph("📄 File Name", styles, bold=True), auto_paragraph(file_name, styles)],
        [auto_paragraph("📅 Generated On", styles, bold=True), auto_paragraph(ts, styles)],
        [auto_paragraph("💼 Applied Role", styles, bold=True), auto_paragraph(applied_role, styles)],
        [auto_paragraph("⏱️ Processing Time", styles, bold=True), auto_paragraph(proc_time, styles)],
        [auto_paragraph("🌐 Primary Language", styles, bold=True), auto_paragraph(primary_language, styles)],
        [auto_paragraph("🔧 Analysis Version", styles, bold=True), auto_paragraph(analysis_version, styles)],
        [auto_paragraph("📋 Analysis Type", styles, bold=True), auto_paragraph(analysis_type.replace('_', ' ').title(), styles)],
    ]
    
    meta_table = Table(meta_data, colWidths=[180, CONTENT_WIDTH - 220])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#FFFFFF")),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ('BOX', (0,0), (-1,-1), 1.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0,0), (-1,-1), TEXT),
        ('FONTNAME', (0,0), (-1,-1), 'noto_regular'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    
    story.append(Spacer(1, 40))
    footer_style = ParagraphStyle(
        name='FooterNote',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=HexColor("#7F8C8D"),
        alignment=TA_CENTER
    )
    footer_note = Paragraph(
        "This report was automatically generated using advanced AI analysis for recruitment interviews",
        footer_style
    )
    story.append(footer_note)
    story.append(Spacer(1, 30))
    
    return story

def create_pdf_from_json(data, out_pdf_path):
    """Create PDF report from JSON data"""
    try:
        if out_pdf_path is None:
            temp_dir = tempfile.gettempdir()
            out_pdf_path = os.path.join(temp_dir, "convlyze_hr_interview_report.pdf")
        
        if data is None:
            print("Warning: No data provided, creating empty report")
            data = {}
        
        data = validate_data_structure(data)
        
        register_fonts()
        styles = build_styles()
        
        doc = SimpleDocTemplate(
            out_pdf_path,
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN
        )
        story = []

        try:
            story.extend(title_page(data, styles))
            story.append(PageBreak())
        except Exception as e:
            print(f"Error creating title page: {e}")
            story.append(Paragraph("HR Interview Analysis Report", styles['ReportTitle']))
            story.append(Spacer(1, 20))
            story.append(PageBreak())
        
        analysis = safe_get_nested(data, 'analysis_results', default={})
        
        if has_content(analysis):
            story.append(executive_heading("Interview Analysis Results", styles))
            story.append(Spacer(1, 20))
        

        hiring_rec = safe_get_nested(analysis, 'hiring_recommendation', default={}) or \
                     safe_get_nested(analysis, 'candidate_assessment', 'overall_recommendation', default={})
        
        if has_content(hiring_rec):
            story.append(section_heading("Executive Summary - Hiring Recommendation", styles, bg_color=PRIMARY))
            story.append(Spacer(1, 15))
            
            recommendation = safe_string_value(hiring_rec.get('recommendation', ''), 'NOT_ANALYZED').upper()
            rec_score = safe_numeric_value(hiring_rec.get('score', 0), 0)
            confidence = safe_string_value(hiring_rec.get('confidence_level', ''), 'Unknown')
            
            rec_color = get_recommendation_color(recommendation)
            rec_display = recommendation.replace('_', ' ')
            
            rec_para = auto_paragraph(
                f'<font color="white" size="16"><b>Recommendation: {rec_display}</b></font>',
                styles,
                base_style='CustomBody'
            )
            rec_para.style.alignment = TA_CENTER
            
            rec_banner = Table([[rec_para]], colWidths=[CONTENT_WIDTH])
            rec_banner.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), rec_color),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 20),
                ('BOTTOMPADDING', (0,0), (-1,-1), 20),
                ('BOX', (0,0), (-1,-1), 2, HexColor("#000000")),
            ]))
            story.append(rec_banner)
            story.append(Spacer(1, 15))
            
            summary_data = [
                [auto_paragraph("Overall Score", styles, bold=True), auto_paragraph(f"{rec_score:.0f}%", styles)],
                [auto_paragraph("Confidence Level", styles, bold=True), auto_paragraph(confidence, styles)],
            ]
            
            summary_table = Table(summary_data, colWidths=[180, CONTENT_WIDTH - 220])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 15))
            
            strengths = safe_list_value(hiring_rec.get('key_strengths', []))
            if strengths:
                story.append(auto_paragraph("✓ Key Strengths", styles, base_style='SubHeading'))
                for strength in strengths:
                    s = safe_string_value(strength, "")
                    if s:
                        story.append(auto_paragraph(f"• {s}", styles))
                story.append(Spacer(1, 10))
            
            concerns = safe_list_value(hiring_rec.get('key_concerns', []))
            if concerns:
                story.append(auto_paragraph("⚠ Key Concerns", styles, base_style='SubHeading'))
                for concern in concerns:
                    c = safe_string_value(concern, "")
                    if c:
                        story.append(auto_paragraph(f"• {c}", styles))
                story.append(Spacer(1, 10))
            
            red_flags = safe_list_value(hiring_rec.get('red_flags', []))
            if red_flags:
                story.append(auto_paragraph("🚩 Red Flags", styles, base_style='SubHeading'))
                for flag in red_flags:
                    f = safe_string_value(flag, "")
                    if f:
                        story.append(auto_paragraph(f"• {f}", styles))
                story.append(Spacer(1, 10))
            
            next_steps = safe_list_value(hiring_rec.get('next_steps', []))
            if next_steps:
                story.append(auto_paragraph("→ Recommended Next Steps", styles, base_style='SubHeading'))
                for step in next_steps:
                    s = safe_string_value(step, "")
                    if s:
                        story.append(auto_paragraph(f"• {s}", styles))
                story.append(Spacer(1, 20))
        

        candidate_assessment = safe_get_nested(analysis, 'candidate_assessment', default={})
        
        if has_content(candidate_assessment):
            story.append(section_heading("Candidate Assessment", styles))
            story.append(Spacer(1, 15))
            
            assessment_scores = {}
            
            comm_skills = safe_get_nested(candidate_assessment, 'communication_skills', default={})
            if has_content(comm_skills):
                comm_score = safe_numeric_value(comm_skills.get('score', 0), 0)
                assessment_scores['Communication'] = comm_score
                
                story.append(auto_paragraph("Communication Skills", styles, base_style='SubHeading'))
                story.append(auto_paragraph(f"<b>Score:</b> {comm_score:.0f}%", styles))
                
                evidence = safe_list_value(comm_skills.get('evidence', []))
                if evidence:
                    story.append(auto_paragraph("<b>Evidence:</b>", styles))
                    for e in evidence[:3]:
                        story.append(auto_paragraph(f"• {safe_string_value(e)}", styles))
                
                strengths = safe_list_value(comm_skills.get('strengths', []))
                if strengths:
                    strengths_str = ", ".join([safe_string_value(s) for s in strengths])
                    story.append(auto_paragraph(f"<b>Strengths:</b> {strengths_str}", styles))
                
                weaknesses = safe_list_value(comm_skills.get('weaknesses', []))
                if weaknesses:
                    weak_str = ", ".join([safe_string_value(w) for w in weaknesses])
                    story.append(auto_paragraph(f"<b>Areas for Improvement:</b> {weak_str}", styles))
                
                story.append(Spacer(1, 12))
            
            tech_comp = safe_get_nested(candidate_assessment, 'technical_competency', default={})
            if has_content(tech_comp):
                tech_score = safe_numeric_value(tech_comp.get('score', 0), 0)
                assessment_scores['Technical'] = tech_score
                
                story.append(auto_paragraph("Technical Competency", styles, base_style='SubHeading'))
                story.append(auto_paragraph(f"<b>Score:</b> {tech_score:.0f}%", styles))
                
                evidence = safe_list_value(tech_comp.get('evidence', []))
                if evidence:
                    story.append(auto_paragraph("<b>Evidence:</b>", styles))
                    for e in evidence[:3]:
                        story.append(auto_paragraph(f"• {safe_string_value(e)}", styles))
                
                skill_areas = safe_get_nested(tech_comp, 'skill_areas', default={})
                if isinstance(skill_areas, dict) and skill_areas:
                    story.append(auto_paragraph("<b>Skill Areas:</b>", styles))
                    skill_data = []
                    for skill, score in skill_areas.items():
                        skill_name = safe_string_value(skill, "").replace('_', ' ').title()
                        skill_score = safe_numeric_value(score, 0)
                        skill_data.append([
                            auto_paragraph(skill_name, styles),
                            auto_paragraph(f"{skill_score:.0f}%", styles)
                        ])
                    
                    if skill_data:
                        skill_table = Table(skill_data, colWidths=[150, 80])
                        skill_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                            ('BOX', (0,0), (-1,-1), 0.5, BORDER),
                            ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                            ('LEFTPADDING', (0,0), (-1,-1), 8),
                            ('TOPPADDING', (0,0), (-1,-1), 4),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                        ]))
                        story.append(skill_table)
                
                story.append(Spacer(1, 12))
            
            cultural_fit = safe_get_nested(candidate_assessment, 'cultural_fit', default={})
            if has_content(cultural_fit):
                cultural_score = safe_numeric_value(cultural_fit.get('score', 0), 0)
                assessment_scores['Cultural Fit'] = cultural_score
                
                story.append(auto_paragraph("Cultural Fit", styles, base_style='SubHeading'))
                story.append(auto_paragraph(f"<b>Score:</b> {cultural_score:.0f}%", styles))
                
                alignment = safe_list_value(cultural_fit.get('alignment_indicators', []))
                if alignment:
                    align_str = ", ".join([safe_string_value(a) for a in alignment])
                    story.append(auto_paragraph(f"<b>Alignment Indicators:</b> {align_str}", styles))
                
                company_research = safe_string_value(cultural_fit.get('company_research', ''))
                if company_research:
                    story.append(auto_paragraph(f"<b>Company Research:</b> {company_research}", styles))
                
                story.append(Spacer(1, 12))
            
            if assessment_scores:
                try:
                    chart_path = os.path.join(tempfile.gettempdir(), "assessment_radar.png")
                    safe_radar_chart(assessment_scores, chart_path, "Candidate Assessment Overview")
                    radar_img = Image(chart_path, width=300, height=300)
                    radar_img.hAlign = 'CENTER'
                    story.append(radar_img)
                    story.append(Spacer(1, 20))
                except Exception as e:
                    print(f"Error creating assessment radar: {e}")
        
        interviewer_quality = safe_get_nested(analysis, 'interviewer_quality', default={})
        
        if has_content(interviewer_quality):
            story.append(section_heading("Interviewer Quality Assessment", styles))
            story.append(Spacer(1, 15))
            
            int_structure = safe_numeric_value(interviewer_quality.get('interview_structure_score', 0), 0)
            engagement = safe_numeric_value(interviewer_quality.get('engagement_level', 0), 0)
            
            int_data = [
                [auto_paragraph("Interview Structure Score", styles, bold=True), auto_paragraph(f"{int_structure:.0f}%", styles)],
                [auto_paragraph("Engagement Level", styles, bold=True), auto_paragraph(f"{engagement:.0f}%", styles)],
            ]
            
            question_quality = safe_string_value(interviewer_quality.get('question_quality', ''))
            if question_quality:
                int_data.append([
                    auto_paragraph("Question Quality", styles, bold=True),
                    auto_paragraph(question_quality, styles)
                ])
            
            int_table = Table(int_data, colWidths=[200, CONTENT_WIDTH - 240])
            int_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(int_table)
            story.append(Spacer(1, 12))
            
            improvements = safe_list_value(interviewer_quality.get('areas_for_improvement', []))
            if improvements:
                story.append(auto_paragraph("Areas for Improvement", styles, base_style='SubHeading'))
                for imp in improvements:
                    story.append(auto_paragraph(f"• {safe_string_value(imp)}", styles))
                story.append(Spacer(1, 20))
        
        overall_score = safe_numeric_value(analysis.get('overall_interview_score', 0), 0)
        if overall_score > 0:
            story.append(section_heading("Overall Interview Score", styles))
            story.append(Spacer(1, 15))
            
            score_para = auto_paragraph(
                f'<font size="24" color="{PRIMARY}"><b>{overall_score:.0f}%</b></font>',
                styles
            )
            score_para.style.alignment = TA_CENTER
            story.append(score_para)
            story.append(Spacer(1, 20))
        
        summary_data = safe_get_nested(analysis, 'summary', default={})
        if has_content(summary_data):
            story.append(section_heading("Interview Summary", styles))
            story.append(Spacer(1, 10))
            
            summary_text = safe_string_value(summary_data.get('summary', ''))
            if summary_text:
                story.append(auto_paragraph(summary_text, styles))
                story.append(Spacer(1, 10))
            
            key_points = safe_list_value(summary_data.get('key_points', []))
            if key_points:
                story.append(auto_paragraph("Key Discussion Points", styles, base_style='SubHeading'))
                for point in key_points:
                    story.append(auto_paragraph(f"• {safe_string_value(point)}", styles))
                story.append(Spacer(1, 10))
            
            position = safe_string_value(summary_data.get('position_discussed', ''))
            if position:
                story.append(auto_paragraph(f"<b>Position Discussed:</b> {position}", styles))
                story.append(Spacer(1, 20))
        
        sentiment = safe_get_nested(analysis, 'sentiment', default={})
        
        if has_content(sentiment):
            story.append(section_heading("Sentiment Analysis", styles))
            story.append(Spacer(1, 10))
            
            overall_sentiment = safe_string_value(sentiment.get('overall_sentiment', ''), 'neutral').title()
            sentiment_score = safe_numeric_value(sentiment.get('sentiment_score', 0), 0)
            
            sent_data = [
                [auto_paragraph("Overall Sentiment", styles, bold=True), auto_paragraph(overall_sentiment, styles)],
                [auto_paragraph("Sentiment Score (Hiring Confidence)", styles, bold=True), auto_paragraph(f"{sentiment_score:.0f}%", styles)],
            ]
            
            sent_table = Table(sent_data, colWidths=[220, CONTENT_WIDTH - 260])
            sent_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                ('BOX', (0,0), (-1,-1), 1, BORDER),
                ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(sent_table)
            story.append(Spacer(1, 12))
            
            emotional_tones = safe_list_value(sentiment.get('emotional_tones', []))
            if emotional_tones:
                tones_str = ", ".join([safe_string_value(t).title() for t in emotional_tones])
                story.append(auto_paragraph(f"<b>Emotional Tones:</b> {tones_str}", styles))
                story.append(Spacer(1, 20))
        
        quality_scores = safe_get_nested(analysis, 'quality_scores', default={})
        
        if has_content(quality_scores):
            story.append(section_heading("Audio & Transcription Quality", styles))
            story.append(Spacer(1, 10))
            
            quality_data = []
            
            audio_quality = safe_numeric_value(quality_scores.get('audio_quality_score', 0), 0)
            if audio_quality > 0:
                quality_data.append([
                    auto_paragraph("Audio Quality", styles, bold=True),
                    auto_paragraph(f"{audio_quality:.0f}%", styles)
                ])
            
            speech_clarity = safe_numeric_value(quality_scores.get('speech_clarity_score', 0), 0)
            if speech_clarity > 0:
                quality_data.append([
                    auto_paragraph("Speech Clarity", styles, bold=True),
                    auto_paragraph(f"{speech_clarity:.0f}%", styles)
                ])
            
            transcription_conf = safe_numeric_value(quality_scores.get('transcription_confidence', 0), 0)
            if transcription_conf > 0:
                quality_data.append([
                    auto_paragraph("Transcription Confidence", styles, bold=True),
                    auto_paragraph(f"{transcription_conf:.0f}%", styles)
                ])
            
            analysis_reliability = safe_numeric_value(quality_scores.get('analysis_reliability', 0), 0)
            if analysis_reliability > 0:
                quality_data.append([
                    auto_paragraph("Analysis Reliability", styles, bold=True),
                    auto_paragraph(f"{analysis_reliability:.0f}%", styles)
                ])
            
            overall_confidence = safe_numeric_value(quality_scores.get('overall_confidence', 0), 0)
            if overall_confidence > 0:
                quality_data.append([
                    auto_paragraph("Overall Confidence", styles, bold=True),
                    auto_paragraph(f"{overall_confidence:.0f}%", styles)
                ])
            
            if quality_data:
                quality_table = Table(quality_data, colWidths=[200, CONTENT_WIDTH - 240])
                quality_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                    ('BOX', (0,0), (-1,-1), 1, BORDER),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                    ('LEFTPADDING', (0,0), (-1,-1), 10),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(quality_table)
                story.append(Spacer(1, 20))

        speakers_info = safe_get_nested(analysis, 'speakers', default={})
        
        if has_content(speakers_info):
            story.append(section_heading("Speaker Analysis", styles))
            story.append(Spacer(1, 10))
            
            speaker_count = safe_numeric_value(speakers_info.get('speaker_count', 0), 0)
            interaction = safe_string_value(speakers_info.get('interaction_pattern', ''), '-')
            
            story.append(auto_paragraph(f"<b>Speaker Count:</b> {speaker_count}", styles))
            story.append(auto_paragraph(f"<b>Interaction Pattern:</b> {interaction}", styles))
            story.append(Spacer(1, 10))
            
            speakers = safe_list_value(speakers_info.get('speakers', []))
            if speakers:
                speaker_rows = [[
                    auto_paragraph("Speaker", styles, bold=True),
                    auto_paragraph("Speaking Time %", styles, bold=True),
                    auto_paragraph("Turn Count", styles, bold=True),
                    auto_paragraph("Characteristics", styles, bold=True),
                ]]
                
                for sp in speakers:
                    if isinstance(sp, dict):
                        speaker_rows.append([
                            auto_paragraph(safe_string_value(sp.get('speaker_name', ''), '-'), styles),
                            auto_paragraph(f"{safe_numeric_value(sp.get('speaking_time_percentage', 0), 0):.0f}%", styles),
                            auto_paragraph(str(safe_numeric_value(sp.get('turn_count', 0), 0)), styles),
                            auto_paragraph(safe_string_value(sp.get('characteristics', ''), '-'), styles),
                        ])
                
                if len(speaker_rows) > 1:
                    speaker_table = Table(speaker_rows, colWidths=[90, 90, 70, CONTENT_WIDTH - 270])
                    speaker_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING', (0,0), (-1,-1), 8),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(speaker_table)
                    story.append(Spacer(1, 20))
        
        topics_data = safe_get_nested(analysis, 'topics', default={})
        
        if has_content(topics_data):
            story.append(section_heading("Discussion Topics", styles))
            story.append(Spacer(1, 10))
            
            categories = safe_list_value(topics_data.get('topic_categories', []))
            if categories:
                cat_str = ", ".join([safe_string_value(c).replace('_', ' ').title() for c in categories])
                story.append(auto_paragraph(f"<b>Topic Categories:</b> {cat_str}", styles))
                story.append(Spacer(1, 10))
            
            key_topics = safe_list_value(topics_data.get('key_topics', []))
            if key_topics:
                topic_rows = [[
                    auto_paragraph("Topic", styles, bold=True),
                    auto_paragraph("Importance", styles, bold=True),
                    auto_paragraph("Mentions", styles, bold=True),
                    auto_paragraph("Context", styles, bold=True),
                ]]
                
                for topic in key_topics:
                    if isinstance(topic, dict):
                        topic_rows.append([
                            auto_paragraph(safe_string_value(topic.get('topic', ''), '-'), styles),
                            auto_paragraph(f"{safe_numeric_value(topic.get('importance_score', 0), 0):.0f}%", styles),
                            auto_paragraph(str(safe_numeric_value(topic.get('mentions', 0), 0)), styles),
                            auto_paragraph(safe_string_value(topic.get('context', ''), '-'), styles),
                        ])
                
                if len(topic_rows) > 1:
                    topic_table = Table(topic_rows, colWidths=[120, 80, 60, CONTENT_WIDTH - 280])
                    topic_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(topic_table)
                    story.append(Spacer(1, 20))

        actions = safe_get_nested(analysis, 'action_items', default={})
        
        if has_content(actions):
            story.append(section_heading("Action Items & Next Steps", styles))
            story.append(Spacer(1, 10))
            
            action_list = safe_list_value(actions.get('action_items', []))
            if action_list:
                action_rows = [[
                    auto_paragraph("Task", styles, bold=True),
                    auto_paragraph("Assignee", styles, bold=True),
                    auto_paragraph("Deadline", styles, bold=True),
                    auto_paragraph("Priority", styles, bold=True),
                ]]
                
                for item in action_list:
                    if isinstance(item, dict):
                        action_rows.append([
                            auto_paragraph(safe_string_value(item.get('task', ''), '-'), styles),
                            auto_paragraph(safe_string_value(item.get('assignee', ''), '-'), styles),
                            auto_paragraph(safe_string_value(item.get('deadline', ''), '-'), styles),
                            auto_paragraph(safe_string_value(item.get('priority', ''), '-').upper(), styles),
                        ])
                
                if len(action_rows) > 1:
                    action_table = Table(action_rows, colWidths=[CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.2, CONTENT_WIDTH*0.2, CONTENT_WIDTH*0.15])
                    action_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(action_table)
                    story.append(Spacer(1, 12))
            
            next_steps = safe_list_value(actions.get('next_steps', []))
            if next_steps:
                story.append(auto_paragraph("Next Steps", styles, base_style='SubHeading'))
                for step in next_steps:
                    story.append(auto_paragraph(f"• {safe_string_value(step)}", styles))
                story.append(Spacer(1, 20))
        
        # TRANSCRIPT
        transcript_data = safe_get_nested(analysis, 'transcript', default={})
        
        if has_content(transcript_data):
            story.append(section_heading("Interview Transcript", styles))
            story.append(Spacer(1, 10))
            
            segments = safe_list_value(transcript_data.get('segments', []))
            if segments:
                trans_rows = [[
                    auto_paragraph("Speaker", styles, bold=True),
                    auto_paragraph("Time", styles, bold=True),
                    auto_paragraph("Text", styles, bold=True),
                ]]
                
                for seg in segments:
                    if isinstance(seg, dict):
                        speaker = safe_string_value(seg.get('speaker_name', ''), '-')
                        time = safe_string_value(seg.get('start_time', ''), '-')
                        text = safe_string_value(seg.get('text', ''), '-')
                        
                        trans_rows.append([
                            auto_paragraph(speaker, styles),
                            auto_paragraph(time, styles),
                            auto_paragraph(text, styles),
                        ])
                
                if len(trans_rows) > 1:
                    trans_table = Table(trans_rows, colWidths=[80, 60, CONTENT_WIDTH - 160])
                    trans_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(trans_table)
        
        try:
            doc.build(story)
            return out_pdf_path
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise
    
    except Exception as e:
        print(f"Critical error in PDF creation: {e}")
        try:
            doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
            story = [
                Paragraph("HR Interview Analysis Report", getSampleStyleSheet()['Title']),
                Spacer(1, 20),
                Paragraph("Error occurred while generating full report.", getSampleStyleSheet()['Normal']),
                Paragraph(f"Error details: {str(e)}", getSampleStyleSheet()['Normal'])
            ]
            doc.build(story)
            return out_pdf_path
        except Exception as fallback_error:
            print(f"Even fallback PDF creation failed: {fallback_error}")
            raise


# ---------------------------------------------------------------------------
# S3 Upload helpers  (replaces the old convlyze API upload)
# ---------------------------------------------------------------------------

def upload_pdf_to_s3(pdf_path: str, filename: str = None) -> str:
    """
    Upload a PDF file to S3 and return its public URL.

    Required env vars:
        AWS_ACCESS_KEY_ID       - AWS access key
        AWS_SECRET_ACCESS_KEY   - AWS secret key
        AWS_REGION              - e.g. "ap-south-1"  (defaults to ap-south-1)
        S3_BUCKET_NAME          - target bucket name
        S3_REPORTS_PREFIX       - key prefix, e.g. "reports/"  (defaults to "reports/")
    """
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region     = os.getenv("AWS_REGION", "ap-south-1")
    bucket_name    = os.getenv("S3_BUCKET_NAME")
    prefix         = os.getenv("S3_REPORTS_PREFIX", "reports/")

    if not all([aws_access_key, aws_secret_key, bucket_name]):
        raise ValueError(
            "Missing S3 configuration. Set AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME in your .env file."
        )

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_report_{timestamp}.pdf"

    s3_key = f"{prefix}{filename}"

    try:
        s3_client = boto3.client(
            "s3",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        s3_client.upload_file(
            pdf_path,
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
        print(f"PDF uploaded to S3: {url}")
        return url

    except (BotoCoreError, ClientError) as e:
        raise Exception(f"S3 upload failed: {e}") from e


def generate_and_upload_report(data):
    """Generate PDF report and upload to S3, returning the S3 URL."""
    try:
        if data is None:
            print("Warning: No data provided for report generation")
            data = {}

        if not isinstance(data, dict):
            print(f"Warning: Input data is not a dict (type: {type(data)}), converting...")
            data = {}

        data = validate_data_structure(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_pdf = os.path.join(
            tempfile.gettempdir(),
            f"convlyze_hr_interview_report_{timestamp}.pdf"
        )

        create_pdf_from_json(data, out_pdf)

        if not os.path.exists(out_pdf):
            raise Exception("PDF file was not created successfully")

        url = upload_pdf_to_s3(out_pdf, filename=os.path.basename(out_pdf))

        # Clean up local temp file after successful upload
        try:
            os.remove(out_pdf)
        except Exception:
            pass

        return url

    except Exception as e:
        print(f"Error in generate_and_upload_report: {e}")
        raise


def create_fallback_report(error_message="Unknown error occurred"):
    """Create a basic fallback report when main report generation fails"""
    return {
        'validation': {
            'is_valid_interview': False,
            'content_type': 'ERROR',
            'reason': error_message
        },
        'metadata': {
            'file_url': 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': 0,
            'primary_language': 'Unknown',
            'analysis_version': '2.0_HR_RECRUITMENT',
            'analysis_type': 'recruitment_interview',
            'status': 'failed'
        },
        'language_info': {
            'primary_language': 'Unknown',
            'multilingual_content': 'unknown'
        },
        'analysis_results': {
            'summary': {
                'summary': f'Report generation failed: {error_message}',
                'key_points': [],
                'main_topics': []
            },
            'transcript': {
                'transcript': '',
                'segments': []
            },
            'candidate_assessment': {
                'overall_recommendation': {
                    'score': 0,
                    'recommendation': 'NOT_ANALYZED',
                    'confidence_level': 'N/A',
                    'key_strengths': [],
                    'key_concerns': [error_message],
                    'red_flags': [],
                    'next_steps': []
                }
            },
            'overall_interview_score': 0
        }
    }

__all__ = [
    'create_pdf_from_json',
    'generate_and_upload_report',
    'upload_pdf_to_s3',
    'validate_data_structure',
    'create_fallback_report',
    'safe_get_nested',
    'safe_numeric_value',
    'safe_list_value',
    'safe_string_value',
    'has_content'
]

if __name__ == "__main__":
    """Test the report generation with sample HR interview data"""
    sample_data = {
        "validation": {
            "is_valid_interview": True,
            "content_type": "VALID_INTERVIEW",
            "validation_details": {
                "speaker_count_estimate": 2,
                "has_interview_structure": True,
                "contains_job_discussion": True,
                "content_duration_sufficient": True
            },
            "reason": "Clear interview conversation between interviewer and candidate",
            "recommendation": "PROCEED"
        },
        "metadata": {
            "processing_time_seconds": 45.2,
            "timestamp": "2024-01-15T10:30:00",
            "primary_language": "english",
            "file_url": "https://example.com/interview.mp3",
            "status": "completed_full_analysis",
            "analysis_version": "2.0_HR_RECRUITMENT",
            "analysis_type": "recruitment_interview"
        },
        "language_info": {
            "primary_language": "english",
            "multilingual_content": "single_language",
            "supported_languages": ["english", "hindi", "spanish", "french"]
        },
        "analysis_results": {
            "summary": {
                "summary": "Strong technical candidate with excellent communication skills.",
                "key_points": ["5 years of Python development experience"],
                "main_topics": ["Python Development", "Team Leadership"],
                "position_discussed": "Senior Python Developer",
                "applied_role": "Senior Python Developer"
            },
            "hiring_recommendation": {
                "score": 80,
                "recommendation": "HIRE",
                "confidence_level": "High",
                "key_strengths": ["Strong technical foundation"],
                "key_concerns": ["Limited cloud experience"],
                "red_flags": [],
                "next_steps": ["Technical round", "Team assessment"]
            },
            "overall_interview_score": 82,
        }
    }

    print("=" * 60)
    print("Testing HR Recruitment Interview Report Generation")
    print("=" * 60)

    try:
        print("\n1. Creating PDF report...")
        pdf_path = create_pdf_from_json(sample_data, "test_hr_recruitment_report.pdf")
        print(f"   ✓ Report generated successfully: {pdf_path}")

        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path) / 1024
            print(f"   ✓ File size: {file_size:.2f} KB")

        print("\n2. Testing fallback report...")
        fallback = create_fallback_report("Test error message")
        print(f"   ✓ Fallback data structure created")

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()