import sys
import os
import json
import tempfile
import requests
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

# Font configuration
FONT_DIR = "font"
FONTS = {
    "noto_regular": os.path.join(FONT_DIR, "NotoSans-Regular.ttf"),
    "noto_bold": os.path.join(FONT_DIR, "NotoSans-Bold.ttf"),
    "noto_deva": os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf"),
    "noto_deva_bold": os.path.join(FONT_DIR, "NotoSansDevanagari-Bold.ttf"),
    "noto_arabic": os.path.join(FONT_DIR, "NotoSansArabic-Regular.ttf"),
    "noto_arabic_bold": os.path.join(FONT_DIR, "NotoSansArabic-Bold.ttf"),
}

# Page layout
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 30
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
CHART_W = 240
CHART_H = 180

# Color palette for exit interviews
PRIMARY = HexColor("#C0392B")  # Deep red for exits
ACCENT = HexColor("#E67E22")   # Orange accent
CARD_BG = HexColor("#FFFFFF")
LIGHT_BG = HexColor("#FDF2F0")
TEXT = HexColor("#222222")
BORDER = HexColor("#DDDDDD")
BRAND_BLUE = HexColor("#2E86C1")
CRITICAL_RED = HexColor("#E74C3C")
SERIOUS_ORANGE = HexColor("#E67E22")
SIGNIFICANT_YELLOW = HexColor("#F39C12")
POSITIVE_GREEN = HexColor("#27AE60")
WARNING_AMBER = HexColor("#D68910")

RL_PALETTE = {
    "neutral_bg": HexColor("#F8F9FA"),
    "border": HexColor("#DDDDDD"),
    "text": HexColor("#222222"),
    "primary": HexColor("#C0392B"),
    "secondary": HexColor("#E67E22"),
    "positive": HexColor("#27AE60"),
    "neutral": HexColor("#FFC107"),
    "negative": HexColor("#E74C3C"),
    "critical": HexColor("#C0392B"),
    "serious": HexColor("#E67E22"),
    "significant": HexColor("#F39C12"),
}

MPL_PALETTE = {
    "neutral_bg": "#F8F9FA",
    "border": "#DDDDDD",
    "text": "#222222",
    "primary": "#C0392B",
    "secondary": "#E67E22",
    "positive": "#27AE60",
    "neutral": "#FFC107",
    "negative": "#E74C3C",
    "critical": "#C0392B",
    "serious": "#E67E22",
    "significant": "#F39C12",
}

REGISTERED_FONTS = {}


def validate_data_structure(data):
    """Validate and sanitize exit interview data structure"""
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
            'quality_scores': {},
            'exit_reasons': {},
            'key_strengths': {},
            'concerns': {},
            'red_flags': {},
            'assessment': {},
            'qa_pairs': {},
            'speakers': {}
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
        ('BOX', (0,0), (-1,-1), 0.75, HexColor("#A93226")),
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


def get_severity_color(severity):
    """Get color based on red flag severity"""
    sev = safe_string_value(severity, "").lower()
    if "critical" in sev:
        return CRITICAL_RED
    elif "serious" in sev:
        return SERIOUS_ORANGE
    elif "significant" in sev:
        return SIGNIFICANT_YELLOW
    return ACCENT


def download_logo_image(logo_url):
    """Download and save logo image"""
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
    """Create Convlyze brand header"""
    logo_url = "https://convlyze-web.s3.ap-south-1.amazonaws.com/uploads/user_images/Oe8h91lUx.png"
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
            return None
    else:
        fallback_style = ParagraphStyle(
            name='FallbackBrand',
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=BRAND_BLUE,
            alignment=TA_CENTER
        )
        return Paragraph("Convlyze - AI Exit Interview Intelligence", fallback_style)


def safe_reason_chart(reasons_data, outpath, title="Reasons for Leaving"):
    """Create horizontal bar chart for exit reasons"""
    try:
        if not reasons_data or not isinstance(reasons_data, dict):
            reasons_data = {"No Data": 50}
        
        labels = list(reasons_data.keys())
        values = [safe_numeric_value(v, 50) for v in reasons_data.values()]
        
        if not labels:
            labels, values = ["No Data"], [50]
        
        fig = plt.figure(figsize=(6, 4), dpi=150)
        ax = fig.add_subplot(111)
        
        colors = [MPL_PALETTE["negative"] if v > 70 else MPL_PALETTE["neutral"] if v > 40 else MPL_PALETTE["positive"] for v in values]
        
        ax.barh(labels, values, color=colors, alpha=0.8)
        ax.set_xlabel('Impact Score', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        
        for i, v in enumerate(values):
            ax.text(v + 2, i, f'{v:.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating reason chart: {e}")
        fig = plt.figure(figsize=(6, 4), dpi=150)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Chart data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)


def title_page(data, styles):
    """Create title page for exit interview report"""
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
        fallback_style = styles['ReportTitle'].clone('FallbackBrand')
        fallback_style.textColor = BRAND_BLUE
        fallback_style.fontSize = 24
        story.append(Paragraph("Convlyze - AI Exit Interview Intelligence", fallback_style))
        story.append(Spacer(1, 30))
    
    # Title banner
    title_style = styles['ReportTitle'].clone('ReportTitle_banner')
    title_style.textColor = HexColor("#FFFFFF")
    title_style.fontSize = 26
    title_style.leading = 30
    
    title_para = Paragraph("Exit Interview Analysis Report", title_style)
    
    title_banner = Table([[title_para]], colWidths=[CONTENT_WIDTH])
    title_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 35),
        ('BOTTOMPADDING', (0,0), (-1,-1), 35),
        ('BOX', (0,0), (-1,-1), 2, HexColor("#A93226")),
    ]))
    story.append(title_banner)
    story.append(Spacer(1, 25))
    
    # Metadata
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
    file_name = safe_string_value(meta.get("file_name", "")) or os.path.basename(urlparse(file_url).path) if file_url else "Unknown File"
    ts = format_timestamp(safe_string_value(meta.get("timestamp", "-")))
    primary_language = safe_string_value(meta.get("primary_language", "Unknown")).title()
    analysis_version = safe_string_value(meta.get("analysis_version", "v1.0"))
    analysis_type = safe_string_value(meta.get("analysis_type", "exit_interview"))
    
    meta_header = Paragraph("Report Details", styles['SubHeading'])
    story.append(meta_header)
    story.append(Spacer(1, 10))
    
    meta_data = [
        [auto_paragraph("📄 File Name", styles, bold=True), auto_paragraph(file_name, styles)],
        [auto_paragraph("📅 Generated On", styles, bold=True), auto_paragraph(ts, styles)],
        [auto_paragraph("⏱️ Processing Time", styles, bold=True), auto_paragraph(proc_time, styles)],
        [auto_paragraph("🌐 Primary Language", styles, bold=True), auto_paragraph(primary_language, styles)],
        [auto_paragraph("🔧 Analysis Version", styles, bold=True), auto_paragraph(analysis_version, styles)],
        [auto_paragraph("📋 Analysis Type", styles, bold=True), auto_paragraph(analysis_type.replace('_', ' ').title(), styles)],
    ]
    
    meta_table = Table(meta_data, colWidths=[180, CONTENT_WIDTH - 220])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#FFFFFF")),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [HexColor("#FFFFFF"), HexColor("#FDF2F0")]),
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
    story.append(Spacer(1, 25))
    
    # EMPLOYEE DETAILS - MOVED HERE FROM EXIT INTERVIEW ASSESSMENT
    analysis = safe_get_nested(data, 'analysis_results', default={})
    assessment_data = safe_get_nested(analysis, 'assessment', default={})
    employee_profile = safe_get_nested(assessment_data, 'employee_profile', default={})
    
    if has_content(employee_profile):
        employee_header = Paragraph("Employee Details", styles['SubHeading'])
        story.append(employee_header)
        story.append(Spacer(1, 10))
        
        profile_data = []
        
        role = safe_string_value(employee_profile.get('role', ''), '')
        if role:
            profile_data.append([auto_paragraph("👤 Role", styles, bold=True), auto_paragraph(role, styles)])
        
        tenure = safe_numeric_value(employee_profile.get('tenure_years', 0), 0)
        if tenure > 0:
            profile_data.append([auto_paragraph("📅 Tenure", styles, bold=True), auto_paragraph(f"{tenure:.1f} years", styles)])
        
        dept = safe_string_value(employee_profile.get('department', ''), '')
        if dept:
            profile_data.append([auto_paragraph("🏢 Department", styles, bold=True), auto_paragraph(dept, styles)])
        
        perf_level = safe_string_value(employee_profile.get('performance_level', ''), '')
        if perf_level:
            profile_data.append([auto_paragraph("⭐ Performance Level", styles, bold=True), auto_paragraph(perf_level, styles)])
        
        rehire = safe_get_nested(employee_profile, 'rehire_eligible', default=None)
        if rehire is not None:
            rehire_text = "YES" if rehire else "NO"
            profile_data.append([auto_paragraph("🔄 Rehire Eligible", styles, bold=True), auto_paragraph(rehire_text, styles)])
        
        if profile_data:
            profile_table = Table(profile_data, colWidths=[180, CONTENT_WIDTH - 220])
            profile_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), HexColor("#FFFFFF")),
                ('ROWBACKGROUNDS', (0,0), (-1,-1), [HexColor("#FFFFFF"), HexColor("#FDF2F0")]),
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
            story.append(profile_table)
            story.append(Spacer(1, 20))
    
    story.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        name='FooterNote',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=HexColor("#7F8C8D"),
        alignment=TA_CENTER
    )
    footer_note = Paragraph(
        "This report was automatically generated using advanced AI analysis for exit interviews",
        footer_style
    )
    story.append(footer_note)
    story.append(Spacer(1, 30))
    
    return story


def create_exit_pdf_from_json(data, out_pdf_path):
    """Create exit interview PDF report from JSON data"""
    try:
        if out_pdf_path is None:
            temp_dir = tempfile.gettempdir()
            out_pdf_path = os.path.join(temp_dir, "convlyze_exit_interview_report.pdf")
        
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
        
        # Title page
        try:
            story.extend(title_page(data, styles))
            story.append(PageBreak())
        except Exception as e:
            print(f"Error creating title page: {e}")
            story.append(Paragraph("Exit Interview Analysis Report", styles['ReportTitle']))
            story.append(Spacer(1, 20))
            story.append(PageBreak())
        
        analysis = safe_get_nested(data, 'analysis_results', default={})
        
        if has_content(analysis):
            story.append(executive_heading("Exit Interview Analysis Results", styles))
            story.append(Spacer(1, 20))
        
        # EXECUTIVE SUMMARY (without Recommended Immediate Actions)
        summary_data = safe_get_nested(analysis, 'summary', default={})
        if has_content(summary_data):
            story.append(section_heading("Executive Summary", styles, bg_color=PRIMARY))
            story.append(Spacer(1, 15))
            
            executive_summary = safe_string_value(summary_data.get('executive_summary', ''))
            if executive_summary:
                story.append(auto_paragraph(executive_summary, styles))
                story.append(Spacer(1, 12))
            
            key_takeaways = safe_list_value(summary_data.get('key_takeaways', []))
            if key_takeaways:
                story.append(auto_paragraph("Key Takeaways", styles, base_style='SubHeading'))
                for takeaway in key_takeaways:
                    story.append(auto_paragraph(f"• {safe_string_value(takeaway)}", styles))
                story.append(Spacer(1, 12))
            
            strategic_implications = safe_string_value(summary_data.get('strategic_implications', ''))
            if strategic_implications:
                story.append(auto_paragraph("Strategic Implications", styles, base_style='SubHeading'))
                story.append(auto_paragraph(strategic_implications, styles))
                story.append(Spacer(1, 12))
            
            overall_outcome = safe_string_value(summary_data.get('overall_outcome', ''))
            if overall_outcome:
                story.append(auto_paragraph("Overall Outcome", styles, base_style='SubHeading'))
                story.append(auto_paragraph(overall_outcome, styles))
                story.append(Spacer(1, 20))
        
        # RED FLAGS - SIMPLIFIED (removed critical table, risk score, potential implications, immediate actions, affects others)
        red_flags_data = safe_get_nested(analysis, 'red_flags', default={})
        
        if has_content(red_flags_data):
            story.append(section_heading("Red Flags Assessment", styles, bg_color=PRIMARY))
            story.append(Spacer(1, 15))
            
            # Red flags summary only
            summary = safe_string_value(red_flags_data.get('red_flags_summary', ''))
            if summary:
                story.append(auto_paragraph(summary, styles))
                story.append(Spacer(1, 12))
            
            # Simplified red flags list (only category, flag, severity, description)
            red_flags_list = safe_list_value(red_flags_data.get('red_flags', []))
            if red_flags_list:
                story.append(auto_paragraph("Identified Red Flags", styles, base_style='SubHeading'))
                story.append(Spacer(1, 10))
                
                for i, flag in enumerate(red_flags_list, 1):
                    if isinstance(flag, dict):
                        category = safe_string_value(flag.get('category', ''), 'General')
                        flag_desc = safe_string_value(flag.get('flag', ''), '')
                        severity = safe_string_value(flag.get('severity', ''), 'medium')
                        description = safe_string_value(flag.get('description', ''), '')
                        
                        severity_color = get_severity_color(severity)
                        
                        # Flag header
                        flag_header = auto_paragraph(
                            f'<font color="white"><b>Red Flag #{i}: {flag_desc}</b></font>',
                            styles
                        )
                        flag_header_table = Table([[flag_header]], colWidths=[CONTENT_WIDTH])
                        flag_header_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,-1), severity_color),
                            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                            ('LEFTPADDING', (0,0), (-1,-1), 10),
                            ('TOPPADDING', (0,0), (-1,-1), 6),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                        ]))
                        story.append(flag_header_table)
                        story.append(Spacer(1, 8))
                        
                        # Simplified flag details (removed risk_score)
                        flag_details = [
                            [auto_paragraph("Category", styles, bold=True), auto_paragraph(category, styles)],
                            [auto_paragraph("Severity", styles, bold=True), auto_paragraph(severity.upper(), styles)],
                        ]
                        
                        flag_table = Table(flag_details, colWidths=[120, CONTENT_WIDTH - 160])
                        flag_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
                            ('BOX', (0,0), (-1,-1), 1, BORDER),
                            ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                            ('LEFTPADDING', (0,0), (-1,-1), 8),
                            ('TOPPADDING', (0,0), (-1,-1), 5),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                        ]))
                        story.append(flag_table)
                        story.append(Spacer(1, 8))
                        
                        if description:
                            story.append(auto_paragraph(f"<b>Description:</b> {description}", styles))
                            story.append(Spacer(1, 6))
                        
                        # Supporting evidence only
                        evidence = safe_list_value(flag.get('supporting_evidence', []))
                        if evidence:
                            story.append(auto_paragraph("<b>Supporting Evidence:</b>", styles))
                            for ev in evidence[:5]:
                                story.append(auto_paragraph(f"  • {safe_string_value(ev)}", styles))
                            story.append(Spacer(1, 15))
                        else:
                            story.append(Spacer(1, 15))
        
        # REASONS FOR LEAVING (removed preventability assessment)
        exit_reasons = safe_get_nested(analysis, 'exit_reasons', default={})
        
        if has_content(exit_reasons):
            story.append(section_heading("Reasons for Leaving", styles))
            story.append(Spacer(1, 15))
            
            departure_type = safe_string_value(exit_reasons.get('departure_type', ''), 'Unknown').title()
            story.append(auto_paragraph(f"<b>Departure Type:</b> {departure_type}", styles))
            story.append(Spacer(1, 10))
            
            # Primary reason
            primary_reason = safe_get_nested(exit_reasons, 'primary_reason', default={})
            if has_content(primary_reason):
                category = safe_string_value(primary_reason.get('category', ''), '')
                subcategory = safe_string_value(primary_reason.get('subcategory', ''), '')
                impact = safe_numeric_value(primary_reason.get('impact_score', 0), 0)
                description = safe_string_value(primary_reason.get('description', ''), '')
                
                story.append(auto_paragraph("Primary Reason", styles, base_style='SubHeading'))
                
                primary_data = [
                    [auto_paragraph("Category", styles, bold=True), auto_paragraph(category, styles)],
                    [auto_paragraph("Specific Issue", styles, bold=True), auto_paragraph(subcategory, styles)],
                    [auto_paragraph("Impact Score", styles, bold=True), auto_paragraph(f"{impact:.0f}%", styles)],
                ]
                
                primary_table = Table(primary_data, colWidths=[150, CONTENT_WIDTH - 190])
                primary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), HexColor("#FADBD8")),
                    ('BOX', (0,0), (-1,-1), 1.5, PRIMARY),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                    ('LEFTPADDING', (0,0), (-1,-1), 10),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(primary_table)
                story.append(Spacer(1, 8))
                
                if description:
                    story.append(auto_paragraph(f"<b>Description:</b> {description}", styles))
                    story.append(Spacer(1, 8))
                
                quotes = safe_list_value(primary_reason.get('employee_quotes', []))
                if quotes:
                    story.append(auto_paragraph("<b>Employee Quotes:</b>", styles))
                    for quote in quotes[:3]:
                        story.append(auto_paragraph(f'  "{safe_string_value(quote)}"', styles))
                    story.append(Spacer(1, 12))
            
            # Contributing reasons
            contributing = safe_list_value(exit_reasons.get('contributing_reasons', []))
            if contributing:
                story.append(auto_paragraph("Contributing Reasons", styles, base_style='SubHeading'))
                story.append(Spacer(1, 8))
                
                contrib_rows = [[
                    auto_paragraph("Category", styles, bold=True),
                    auto_paragraph("Issue", styles, bold=True),
                    auto_paragraph("Impact", styles, bold=True),
                ]]
                
                for reason in contributing:
                    if isinstance(reason, dict):
                        cat = safe_string_value(reason.get('category', ''), '')
                        subcat = safe_string_value(reason.get('subcategory', ''), '')
                        impact = safe_numeric_value(reason.get('impact_score', 0), 0)
                        
                        contrib_rows.append([
                            auto_paragraph(cat, styles),
                            auto_paragraph(subcat, styles),
                            auto_paragraph(f"{impact:.0f}%", styles),
                        ])
                
                if len(contrib_rows) > 1:
                    contrib_table = Table(contrib_rows, colWidths=[120, CONTENT_WIDTH - 240, 80])
                    contrib_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('LEFTPADDING', (0,0), (-1,-1), 8),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(contrib_table)
                    story.append(Spacer(1, 12))
            
            # Reason summary
            reason_summary = safe_string_value(exit_reasons.get('reason_summary', ''))
            if reason_summary:
                story.append(auto_paragraph("<b>Summary:</b>", styles))
                story.append(auto_paragraph(reason_summary, styles))
                story.append(Spacer(1, 20))
        
        # KEY STRENGTHS
        strengths_data = safe_get_nested(analysis, 'key_strengths', default={})
        
        if has_content(strengths_data):
            story.append(section_heading("Key Strengths - What Employees Valued", styles, bg_color=POSITIVE_GREEN))
            story.append(Spacer(1, 15))
            
            strengths_summary = safe_string_value(strengths_data.get('strengths_summary', ''))
            if strengths_summary:
                story.append(auto_paragraph(strengths_summary, styles))
                story.append(Spacer(1, 10))
            
            top_3 = safe_list_value(strengths_data.get('top_3_strengths', []))
            if top_3:
                story.append(auto_paragraph("Top 3 Strengths", styles, base_style='SubHeading'))
                for i, strength in enumerate(top_3, 1):
                    story.append(auto_paragraph(f"{i}. {safe_string_value(strength)}", styles, bold=True))
                story.append(Spacer(1, 12))
            
            # Detailed strengths
            key_strengths = safe_list_value(strengths_data.get('key_strengths', []))
            if key_strengths:
                strength_rows = [[
                    auto_paragraph("Category", styles, bold=True),
                    auto_paragraph("Strength", styles, bold=True),
                    auto_paragraph("Importance", styles, bold=True),
                ]]
                
                for strength in key_strengths:
                    if isinstance(strength, dict):
                        cat = safe_string_value(strength.get('category', ''), '')
                        streng = safe_string_value(strength.get('strength', ''), '')
                        importance = safe_numeric_value(strength.get('importance_score', 0), 0)
                        
                        strength_rows.append([
                            auto_paragraph(cat, styles),
                            auto_paragraph(streng, styles),
                            auto_paragraph(f"{importance:.0f}%", styles),
                        ])
                
                if len(strength_rows) > 1:
                    strength_table = Table(strength_rows, colWidths=[100, CONTENT_WIDTH - 180, 60])
                    strength_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), POSITIVE_GREEN),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('BOX', (0,0), (-1,-1), 1, BORDER),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), HexColor("#E8F8F5")]),
                        ('LEFTPADDING', (0,0), (-1,-1), 8),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(strength_table)
                    story.append(Spacer(1, 20))
        
        # CONCERNS SECTION REMOVED - as per client requirements
        
        # EXIT INTERVIEW ASSESSMENT REMOVED - employee details already moved to title page
        
        # SENTIMENT ANALYSIS REMOVED - as per client requirements
        
        # QUALITY SCORES REMOVED - as per client requirements
        
        # Q&A PAIRS
        qa_data = safe_get_nested(analysis, 'qa_pairs', default={})
        
        if has_content(qa_data):
            qa_pairs = safe_list_value(qa_data.get('qa_pairs', []))
            total_qs = safe_numeric_value(qa_data.get('total_questions', 0), 0)
            
            if qa_pairs or total_qs > 0:
                story.append(section_heading("Interview Questions & Answers", styles))
                story.append(Spacer(1, 10))
                
                if total_qs > 0:
                    story.append(auto_paragraph(f"<b>Total Questions Asked:</b> {int(total_qs)}", styles))
                    story.append(Spacer(1, 10))
                
                if qa_pairs:
                    for i, qa in enumerate(qa_pairs[:10], 1):  # Show first 10
                        if isinstance(qa, dict):
                            question = safe_string_value(qa.get('question', ''), '')
                            answer = safe_string_value(qa.get('answer', ''), '')
                            category = safe_string_value(qa.get('category', ''), '')
                            
                            if question:
                                story.append(auto_paragraph(f"<b>Q{i}:</b> {question}", styles))
                                if category:
                                    story.append(auto_paragraph(f"<i>Category: {category}</i>", styles))
                                if answer:
                                    story.append(auto_paragraph(f"<b>A:</b> {answer}", styles))
                                story.append(Spacer(1, 8))
                    
                    if len(qa_pairs) > 10:
                        story.append(auto_paragraph(f"<i>... and {len(qa_pairs) - 10} more questions</i>", styles))
                    story.append(Spacer(1, 20))
        
        # TRANSCRIPT
        transcript_data = safe_get_nested(analysis, 'transcript', default={})
        
        if has_content(transcript_data):
            story.append(section_heading("Complete Interview Transcript", styles))
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
        
        # Build PDF
        try:
            doc.build(story)
            return out_pdf_path
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise
    
    except Exception as e:
        print(f"Critical error in exit PDF creation: {e}")
        try:
            doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
            story = [
                Paragraph("Exit Interview Analysis Report", getSampleStyleSheet()['Title']),
                Spacer(1, 20),
                Paragraph("Error occurred while generating full report.", getSampleStyleSheet()['Normal']),
                Paragraph(f"Error details: {str(e)}", getSampleStyleSheet()['Normal'])
            ]
            doc.build(story)
            return out_pdf_path
        except Exception as fallback_error:
            print(f"Even fallback PDF creation failed: {fallback_error}")
            raise


def generate_and_upload_exit_report(data):
    """Generate exit interview PDF report and upload to API"""
    api_url = "https://api.convlyze.com/api/users/upload"
    
    try:
        if data is None:
            print("Warning: No data provided for exit report generation")
            data = {}
        
        if not isinstance(data, dict):
            print(f"Warning: Input data is not a dict (type: {type(data)}), converting...")
            data = {}
        
        data = validate_data_structure(data)
        out_pdf = os.path.join(tempfile.gettempdir(), "convlyze_exit_interview_report.pdf")
        create_exit_pdf_from_json(data, out_pdf)
        
        if not os.path.exists(out_pdf):
            raise Exception("Exit PDF file was not created successfully")
        
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
        print(f"Error in generate_and_upload_exit_report: {e}")
        raise


__all__ = [
    'create_exit_pdf_from_json',
    'generate_and_upload_exit_report',
    'validate_data_structure',
    'safe_get_nested',
    'safe_numeric_value',
    'safe_list_value',
    'safe_string_value',
    'has_content'
]


# if __name__ == "__main__":
#     #code has upper 10 qna pairs
#     print("Exit Interview Report Generator - Ready")
#     data="data.json"
#     with open(data, 'r') as f:
#         json_data = json.load(f)
#     url=generate_and_upload_exit_report(json_data)
#     print("Report URL:", url)