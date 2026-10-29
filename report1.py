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
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, Flowable
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
ACCENT2 = HexColor("#4A90E2")  
CARD_BG = HexColor("#FFFFFF")
LIGHT_BG = HexColor("#F5F7FA")
TEXT = HexColor("#222222")
BORDER = HexColor("#DDDDDD")
BRAND_BLUE = HexColor("#2E86C1")  # Convlyze brand color

RL_PALETTE = {
    "neutral_bg": HexColor("#F8F9FA"),
    "border": HexColor("#DDDDDD"),
    "text": HexColor("#222222"),
    "primary": HexColor("#1F3A93"),
    "secondary": HexColor("#4A90E2"),
    "positive": HexColor("#4CAF50"),
    "neutral": HexColor("#FFC107"),
    "negative": HexColor("#F44336"),
    "radar": HexColor("#667EEA"),
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
    "radar": "#667EEA",
}

REGISTERED_FONTS = {}

def validate_data_structure(data):
    """
    Validate and sanitize the input data structure.
    Returns a properly structured dictionary with default values.
    """
    if data is None:
        print("Warning: Input data is None, using default structure")
        data = {}
    
    if not isinstance(data, dict):
        print(f"Warning: Input data is not a dict (type: {type(data)}), converting to dict")
        data = {}
    
    # Ensure top-level structure exists
    default_structure = {
        'metadata': {},
        'analysis_results': {
            'summary': {},
            'scores': {},
            'speakers': {},
            'sentiment': {},
            'emotions': {},
            'action_items': {},
            'topics': {},
            'segment_sentiment': {},
            'transcript': {}
        }
    }
    
    # Merge with defaults
    for key, default_value in default_structure.items():
        if key not in data:
            data[key] = default_value
        elif data[key] is None:
            data[key] = default_value
        elif isinstance(default_value, dict) and not isinstance(data[key], dict):
            data[key] = default_value
    
    return data

def safe_get_nested(data, *keys, default=None):
    """
    Safely get nested dictionary values with fallback to default.
    """
    current = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current

def safe_numeric_value(value, default=0):
    """
    Safely convert value to numeric, with fallback to default.
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        if isinstance(value, dict):
            # If it's a dict, return default
            return default
    except (ValueError, TypeError, AttributeError):
        pass
    
    return default

def safe_list_value(value, default=None):
    """
    Safely get list value with fallback to default.
    """
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
    """
    Safely get string value with fallback to default.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    
    return default

def has_content(obj):
    """
    Enhanced content checking with better null handling.
    """
    if obj is None:
        return False
    if isinstance(obj, dict):
        if not obj:  # Empty dict
            return False
        return any(has_content(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        if not obj:  # Empty sequence
            return False
        return any(has_content(v) for v in obj)
    if isinstance(obj, str):
        s = obj.strip()
        return s != "" and s != "-" and s.lower() != "none"
    if isinstance(obj, (int, float, bool)):
        return True
    return True

def append_flowables_if(story, condition, *flowables):
    """
    Append flowables to story if condition is met.
    """
    if condition:
        story.extend(flowables)

def download_logo_image(logo_url):
    """
    Download and save logo image to temporary file.
    Returns the local path if successful, None otherwise.
    """
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

def executive_heading(text, styles, bg_color=PRIMARY):
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

def create_brand_header():
    """Create Convlyze brand header"""
    logo_url = "https://onereap.s3.ap-south-1.amazonaws.com/assets/onetab.aiLogo.png"
    logo_path = download_logo_image(logo_url)
    
    if logo_path and os.path.exists(logo_path):
        try:
            # Get original image dimensions
            pil_img = PILImage.open(logo_path)
            orig_width, orig_height = pil_img.size
            
            # Scale to desired height while maintaining aspect ratio
            desired_height = 60
            aspect_ratio = orig_width / orig_height
            desired_width = desired_height * aspect_ratio
            
            logo_img = Image(logo_path, width=desired_width, height=desired_height)
            logo_img.hAlign = 'CENTER'  # Center the logo
            return logo_img
        except Exception as e:
            print(f"Error creating logo image: {e}")
            return None
    else:
        # Fallback if logo can't be loaded
        fallback_style = ParagraphStyle(
            name='FallbackBrand',
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=BRAND_BLUE,
            alignment=TA_CENTER
        )
        return Paragraph("OneConvo - AI Interview Intelligence", fallback_style)

def title_page(data, styles):
    story = []
    
    # Validate data structure
    data = validate_data_structure(data)
    
    story.append(Spacer(1, 30))
    
    # Add Convlyze brand header
    try:
        brand_header = create_brand_header()
        if brand_header:
            story.append(brand_header)
            story.append(Spacer(1, 40))
    except Exception as e:
        print(f"Error creating brand header: {e}")
        # Fallback to text-only branding
        fallback_style = styles['ReportTitle'].clone('FallbackBrand')
        fallback_style.textColor = BRAND_BLUE
        fallback_style.fontSize = 24
        story.append(Paragraph("Convlyze - AI-Powered Voice Insights", fallback_style))
        story.append(Spacer(1, 30))

    # Enhanced title banner with gradient-like effect
    title_style = styles['ReportTitle'].clone('ReportTitle_banner')
    title_style.textColor = HexColor("#FFFFFF")  
    title_style.fontSize = 26                   
    title_style.leading = 30

    title_para = Paragraph("Audio Analysis Report", title_style)

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

    # Safe access to topics for subtitle
    topics = safe_get_nested(data, 'analysis_results', 'summary', 'main_topics', default=[])
    topics = safe_list_value(topics, [])
    
    if topics and len(topics) > 0:
        subtitle_style = styles['Subtitle'].clone('EnhancedSubtitle')
        subtitle_style.fontSize = 16
        subtitle_style.textColor = HexColor("#34495E")
        story.append(Paragraph(str(topics[0]), subtitle_style))
        story.append(Spacer(1, 40))

    # Enhanced metadata section with improved styling
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
        """Format timestamp for better readability"""
        try:
            if ts_str and ts_str != "-":
                # Try to parse and reformat if it's an ISO timestamp
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
    analysis_version = safe_string_value(meta.get("analysis_version", "v1.0"))
    
    # Add file size if available
    file_size = safe_string_value(meta.get("file_size", ""))
    if not file_size:
        file_size = "Unknown"

    # Create metadata header
    meta_header = Paragraph("Report Details", styles['SubHeading'])
    story.append(meta_header)
    story.append(Spacer(1, 10))

    # Enhanced metadata table with icons (using Unicode symbols)
    meta_data = [
        [auto_paragraph("📄 File Name", styles, bold=True), auto_paragraph(file_name, styles)],
        [auto_paragraph("📅 Generated On", styles, bold=True), auto_paragraph(ts, styles)],
        [auto_paragraph("⏱️ Processing Time", styles, bold=True), auto_paragraph(proc_time, styles)],
        [auto_paragraph("🌐 Primary Language", styles, bold=True), auto_paragraph(primary_language, styles)],
        [auto_paragraph("🔧 Analysis Version", styles, bold=True), auto_paragraph(analysis_version, styles)],
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

    # Add a subtle footer note
    story.append(Spacer(1, 40))
    footer_style = ParagraphStyle(
        name='FooterNote',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=HexColor("#7F8C8D"),
        alignment=TA_CENTER
    )
    footer_note = Paragraph(
        "This report was automatically generated using advanced AI analysis techniques",
        footer_style
    )
    story.append(footer_note)
    story.append(Spacer(1, 30))

    return story

def register_fonts():
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
    """
    Choose appropriate font for text, handling Arabic, Devanagari, and Latin.
    """
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
    """
    Create paragraph with safe text handling.
    """
    if text is None:
        text = ""
    text = str(text)
    
    base = styles[base_style] 
    style = base.clone(f"{base_style}_temp")
    style.fontName = choose_font_for_text(text, bold=bold)
    return Paragraph(text, style)

def section_heading(text, styles, bg_color=PRIMARY):
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
        fontSize=11,
        leading=14,
        textColor=HexColor("#2E2E2E"),
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))

    safe_add(ParagraphStyle(
        name='Deva',
        parent=styles['Normal'],
        fontName=choose_font_for_text("नमस्ते"),  
        fontSize=11,
        leading=16,
        textColor=HexColor("#222222"),
        spaceAfter=6
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

    safe_add(ParagraphStyle(
        name='CardTitle',
        parent=styles['Normal'],
        fontName=choose_font_for_text("Hello", bold=True),
        fontSize=12,
        leading=14,
        textColor=PRIMARY,
        alignment=TA_CENTER
    ))

    return styles

def safe_pie_chart(dist, outpath, title="Sentiment Distribution"):
    """
    Create pie chart with safe data handling.
    """
    try:
        if not dist or not isinstance(dist, dict):
            # Create default chart if no data
            dist = {"neutral": 100}
        
        labels = list(dist.keys())
        values = [safe_numeric_value(dist[k], 0) for k in labels]
        
        # Filter out zero values
        filtered_data = [(l, v) for l, v in zip(labels, values) if v > 0]
        if not filtered_data:
            filtered_data = [("No Data", 100)]
        
        labels, values = zip(*filtered_data)
        
        colors = [MPL_PALETTE.get(k.lower(), MPL_PALETTE["secondary"]) for k in labels]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        # Create empty figure if error occurs
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.text(0.5, 0.5, 'Chart data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def safe_bar_chart(topics, outpath, title="Topics Importance"):
    """
    Create bar chart with safe data handling.
    """
    try:
        if not topics or not isinstance(topics, list):
            topics = [{"topic": "No Data", "importance_score": 0}]
        
        names = [safe_string_value(t.get("topic", ""), "Unknown") for t in topics if isinstance(t, dict)]
        scores = [safe_numeric_value(t.get("importance_score", 0), 0) for t in topics if isinstance(t, dict)]
        
        if not names:
            names, scores = ["No Data"], [0]
            
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        y_pos = range(len(names))
        ax.barh(y_pos, scores, align="center", color=MPL_PALETTE["secondary"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, max(scores) if scores else 1.0))
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        # Create empty figure if error occurs
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        ax.text(0.5, 0.5, 'Chart data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def safe_radar_chart(emotion_intensity, outpath, title="Emotion Intensity"):
    """
    Create radar chart with safe data handling.
    """
    try:
        if not emotion_intensity or not isinstance(emotion_intensity, dict):
            emotion_intensity = {"neutral": 0.5}
        
        # Convert dict values to safe numeric values
        safe_emotions = {}
        for k, v in emotion_intensity.items():
            safe_emotions[safe_string_value(k, "emotion")] = safe_numeric_value(v, 0.5)
        
        labels = list(safe_emotions.keys())
        values = list(safe_emotions.values())
        
        if not labels:
            labels, values = ["neutral"], [0.5]

        N = len(labels)
        values += values[:1]
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]

        fig = plt.figure(figsize=(4, 3), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=MPL_PALETTE["radar"])
        ax.fill(angles, values, alpha=0.25, color=MPL_PALETTE["radar"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(title, y=1.08, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        # Create empty figure if error occurs
        fig = plt.figure(figsize=(4, 3), dpi=150)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Chart data unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)

def safe_transcript_timeline(segments, outpath, title="Transcript Sentiment Timeline"):
    """
    Create timeline chart with safe data handling.
    """
    try:
        if not segments or not isinstance(segments, list):
            return
            
        times, scores, labels, colors = [], [], [], []

        for seg in segments:
            if not isinstance(seg, dict):
                continue
                
            start = safe_numeric_value(seg.get("start_time"), None)
            sentiment = safe_string_value(seg.get("sentiment", ""), "neutral").lower()
            score = safe_numeric_value(seg.get("sentiment_score"), None)
            
            if start is not None and score is not None:
                times.append(float(start))
                scores.append(float(score))
                labels.append(sentiment.title())
                colors.append(MPL_PALETTE.get(sentiment, MPL_PALETTE["secondary"]))
                
        if not times:
            return

        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=150)
        ax.scatter(times, scores, c=colors, s=50, zorder=3)
        ax.plot(times, scores, color=MPL_PALETTE["secondary"], linewidth=1, alpha=0.7)

        for x, y, lbl in zip(times, scores, labels):
            ax.text(x, y + 0.03, lbl, fontsize=7, ha="center")

        ax.set_xlabel("Start Time (s)")
        ax.set_ylabel("Sentiment Score")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", transparent=True)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating timeline chart: {e}")

# Replace the original chart functions with safe versions
save_pie_chart = safe_pie_chart
save_bar_chart = safe_bar_chart
save_radar_chart = safe_radar_chart
save_transcript_timeline = safe_transcript_timeline

def create_pdf_from_json(data, out_pdf_path):
    """
    Enhanced PDF creation with comprehensive error handling.
    """
    try:
        # Validate and sanitize input data
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
            story.append(Paragraph("Audio Analysis Report", styles['ReportTitle']))
            story.append(Spacer(1, 20))
            story.append(PageBreak())

        # Analysis results
        analysis = safe_get_nested(data, 'analysis_results', default={})

        if has_content(analysis):
            story.append(executive_heading("Analysis Results", styles))
            story.append(Spacer(1, 20))

        # Summary section
        summary_data = safe_get_nested(analysis, 'summary', default={})
        if has_content(summary_data):
            story.append(section_heading("Summary", styles))
            story.append(Spacer(1, 10))

            # Confidence score
            conf_score = safe_numeric_value(summary_data.get("confidence_score"), None)
            if conf_score is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {conf_score:.2f}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 6))

            # Main summary
            main_summary = safe_string_value(summary_data.get("summary"))
            if main_summary:
                story.append(auto_paragraph(main_summary, styles, base_style='CustomBody'))
                story.append(Spacer(1, 10))

            # Key points
            key_points = safe_list_value(summary_data.get("key_points", []))
            if key_points:
                story.append(auto_paragraph("Key Points", styles, base_style='SubHeading'))
                for point in key_points:
                    point_str = safe_string_value(point)
                    if point_str:
                        story.append(auto_paragraph(f"• {point_str}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 8))

            # Main topics
            main_topics = safe_list_value(summary_data.get("main_topics", []))
            if main_topics:
                story.append(auto_paragraph("Main Topics", styles, base_style='SubHeading'))
                table_data = []
                for t in main_topics:
                    topic_str = safe_string_value(t)
                    if topic_str:
                        table_data.append([auto_paragraph(topic_str, styles)])
                
                if table_data:
                    topics_table = Table(table_data, colWidths=[CONTENT_WIDTH * 0.95])
                    topics_table.setStyle(TableStyle([
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), HexColor("#F8F9FA")]),
                        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor("#2E2E2E")),
                        ('ALIGN',(0,0),(-1,-1),'LEFT'),
                        ('FONTNAME', (0,0), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, HexColor("#DDDDDD")),
                        ('BOX', (0,0), (-1,-1), 0.25, HexColor("#DDDDDD")),
                        ('LEFTPADDING',(0,0),(-1,-1),6),
                        ('RIGHTPADDING',(0,0),(-1,-1),6),
                        ('TOPPADDING',(0,0),(-1,-1),4),
                        ('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ]))
                    story.append(topics_table)
                story.append(Spacer(1, 8))

            # Duration analysis
            duration_analysis = safe_string_value(summary_data.get("duration_analysis"))
            if duration_analysis:
                story.append(auto_paragraph("Conversation Flow", styles, base_style='SubHeading'))
                story.append(auto_paragraph(duration_analysis, styles, base_style='CustomBody'))
                story.append(Spacer(1, 20))
        #sales section
        sales_data = safe_get_nested(data, 'analysis_results', 'sales_performance', default={}) \
                     or safe_get_nested(data, 'sales_performance', default={})

        if has_content(sales_data):
            story.append(section_heading("Sales Performance", styles))
            story.append(Spacer(1, 10))

            # Overall score & call rating
            overall_score = safe_numeric_value(sales_data.get("overall_sales_score"), None)
            call_rating = safe_string_value(sales_data.get("call_quality_rating"), "")
            bullets = []
            if overall_score is not None:
                bullets.append(f"Overall Sales Score: {overall_score:.0f}%")

            if call_rating:
                bullets.append(f"Call Quality Rating: {call_rating}")
                
            for b in bullets:
                story.append(auto_paragraph(f"• {b}", styles, base_style='CustomBody'))
            story.append(Spacer(1, 12))

            # Sub-sections under sales_performance
            sub_sections = safe_get_nested(sales_data, 'sales_performance', default={})
            if isinstance(sub_sections, dict):
                for sec_name, sec_data in sub_sections.items():
                    if not isinstance(sec_data, dict):
                        continue

                    # Subsection heading
                    sec_title = sec_name.replace("_", " ").title()
                    story.append(auto_paragraph(sec_title, styles, base_style='SubHeading'))

                    # Score
                    sec_score = safe_numeric_value(sec_data.get("score"), None)
                    if sec_score is not None:
                       story.append(auto_paragraph(f"<b>Score:</b> {sec_score:.0f}%", styles, base_style='CustomBody'))

                    if sec_name == "rep_effectiveness" and "objection_handling_quality" in sec_data:
                     ohq = safe_numeric_value(sec_data.get("objection_handling_quality"), None)
                     if ohq is not None:
                      story.append(auto_paragraph(f"<b>Objection Handling Quality:</b> {ohq:.0f}%", styles, base_style='CustomBody'))
                     
                    # Evidence list
                    evidence_list = safe_list_value(sec_data.get("evidence", []))
                    if evidence_list:
                        story.append(auto_paragraph("<b>Evidence:</b>", styles, base_style='CustomBody'))
                        for ev in evidence_list:
                            story.append(auto_paragraph(f"• {safe_string_value(ev)}", styles, base_style='CustomBody'))

                    # Lists of items (moments, signals, strengths, weaknesses, etc.)
                    for key in [
                        "key_moments", "improvement_areas", "buying_signals",
                        "objections_raised", "strengths", "weaknesses",
                        "next_steps_agreed"
                    ]:
                        vals = safe_list_value(sec_data.get(key, []))
                        if vals:
                            title = key.replace("_", " ").title()
                            story.append(auto_paragraph(f"<b>{title}:</b>", styles, base_style='CustomBody'))
                            for v in vals:
                                story.append(auto_paragraph(f"• {safe_string_value(v)}", styles, base_style='CustomBody'))

                    # Extra fields (strings/numbers)
                    for extra_key in ["decision_timeline", "probability_assessment", "timeline_to_decision"]:
                        val = safe_string_value(sec_data.get(extra_key, ""))
                        if val:
                            title = extra_key.replace("_", " ").title()
                            story.append(auto_paragraph(f"<b>{title}:</b> {val}", styles, base_style='CustomBody'))

                    story.append(Spacer(1, 10))

            # Recommendations
            recs = safe_list_value(sales_data.get("recommendations", []))
            if recs:
                story.append(auto_paragraph("Recommendations", styles, base_style='SubHeading'))
                for r in recs:
                    story.append(auto_paragraph(f"• {safe_string_value(r)}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 15))
                
                
        # Scores section with enhanced error handling
       

        scores_data = safe_get_nested(data, 'analysis_results', 'scores', default={}) or safe_get_nested(data, 'scores', default={})

        if scores_data:
            # Handle both list and dict formats
            if isinstance(scores_data, list) and scores_data:
                scores = scores_data[0] if isinstance(scores_data[0], dict) else {}
            elif isinstance(scores_data, dict):
                scores = scores_data
            else:
                scores = {}

            if scores:
                story.append(section_heading("Scores", styles))
                story.append(Spacer(1, 10))

                table_data = []
                for k, v in scores.items():
                    if isinstance(v, dict):
                        # Expand nested dicts
                        for subk, subv in v.items():
                            sub_key_str = safe_string_value(subk, "").replace("_", " ").title()
                            sub_val_str = safe_string_value(subv, "-")
                            table_data.append([
                                auto_paragraph(sub_key_str, styles),
                                auto_paragraph(sub_val_str, styles)
                            ])
                    elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                        # Expand list of dicts (e.g., confidence_scores_by_speaker)
                        story.append(auto_paragraph(k.replace("_", " ").title(), styles, base_style='SubHeading'))
                        for item in v:
                            speaker = safe_string_value(item.get("speaker", "Unknown"))
                            score = safe_numeric_value(item.get("score"), None)
                            row_key = f"{speaker} Confidence"
                            row_val = f"{score:.0f}%" if score is not None else "-"

                            table_data.append([
                                auto_paragraph(row_key, styles),
                                auto_paragraph(row_val, styles)
                            ])
                    else:
                        key_str = safe_string_value(k, "").replace("_", " ").title()
                        score = safe_numeric_value(v, None)
                        val_str = f"{score:.0f}%" if score is not None else safe_string_value(v, "-")

                        table_data.append([
                            auto_paragraph(key_str, styles),
                            auto_paragraph(val_str, styles)
                        ])

                if table_data:
                    score_table = Table(table_data, colWidths=[150, CONTENT_WIDTH - 170])
                    score_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),  
                        ('TEXTCOLOR', (0, 0), (-1, -1), TEXT),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'noto_bold'),  
                        ('FONTNAME', (1, 0), (1, -1), 'noto_regular'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, HexColor("#CCCCCC")),
                        ('BOX', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    story.append(score_table)
                    story.append(Spacer(1, 15))

        # Speakers section with safe data handling
        speakers_info = safe_get_nested(data, 'analysis_results', 'speakers', default={})

        if has_content(speakers_info):
            story.append(section_heading("Speakers", styles))
            story.append(Spacer(1, 10))

            bullets = [
                f"Interaction Pattern: {safe_string_value(speakers_info.get('interaction_pattern'), '-')}",
                f"Speaker Count: {safe_string_value(speakers_info.get('speaker_count'), '-')}"
            ]
            for b in bullets:
                story.append(auto_paragraph(f"• {b}", styles, base_style='CustomBody'))

            story.append(Spacer(1, 10))

            speaker_rows = [[
                auto_paragraph("Speaker Name", styles, bold=True),
                auto_paragraph("Speaking %", styles, bold=True),
                auto_paragraph("Turn Count", styles, bold=True),
                auto_paragraph("Characteristics", styles, bold=True),
            ]]

            speakers_list = safe_list_value(speakers_info.get("speakers"), [])
            for sp in speakers_list:
                if isinstance(sp, dict):
                    speaker_rows.append([
                        auto_paragraph(safe_string_value(sp.get("speaker_name") or sp.get("speaker_id"), "-"), styles),
                        auto_paragraph(safe_string_value(sp.get("speaking_time_percentage"), "-"), styles),
                        auto_paragraph(safe_string_value(sp.get("turn_count"), "-"), styles),
                        auto_paragraph(safe_string_value(sp.get("characteristics"), "-"), styles),
                    ])

            if len(speaker_rows) > 1:  # Only create table if we have data rows
                speaker_table = Table(speaker_rows, colWidths=[90, 90, 90, CONTENT_WIDTH - 300])
                speaker_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), ACCENT2),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ('FONTNAME', (0, 0), (-1, 0), 'noto_bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 1), (-1, -1), 'noto_regular'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, HexColor("#CCCCCC")),
                    ('BOX', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                    ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F8F9FA")),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(speaker_table)
            story.append(Spacer(1, 15))

        # Sentiment section with enhanced error handling
        sentiment = safe_get_nested(data, 'analysis_results', 'sentiment', default={})

        if has_content(sentiment):
            story.append(section_heading("Sentiment", styles))
            story.append(Spacer(1, 10))

            bullets = [
                f"Overall Sentiment: {safe_string_value(sentiment.get('overall_sentiment'), '-')}",
                f"Sentiment Score: {safe_numeric_value(sentiment.get('sentiment_score'), 0):.0f}%",
               
            ]

            for b in bullets:
                story.append(auto_paragraph(f"• {b}", styles, base_style='CustomBody'))
            story.append(Spacer(1, 8))

            # Sentiment distribution with safe chart creation
            dist = sentiment.get("sentiment_distribution", {})
            if isinstance(dist, dict) and dist:
                dist_rows = [[
                    auto_paragraph("Sentiment", styles, bold=True),
                    auto_paragraph("Percentage", styles, bold=True),
                ]]
                for k, v in dist.items():
                    key_str = safe_string_value(k, "").title()
                    val_str = f"{safe_numeric_value(v, 0):.0f}%"
                    dist_rows.append([
                        auto_paragraph(key_str, styles),
                        auto_paragraph(val_str, styles)
                    ])

                dist_table = Table(dist_rows, colWidths=[120, 100])
                dist_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), ACCENT2),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ('FONTNAME', (0, 0), (-1, 0), 'noto_bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 1), (-1, -1), 'noto_regular'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, BORDER),
                    ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
                    ('BACKGROUND', (0, 1), (-1, -1), RL_PALETTE["neutral_bg"]),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))

                # Safe chart creation
                try:
                    chart_path = os.path.join(tempfile.gettempdir(), "sentiment_pie.png")
                    save_pie_chart(dist, chart_path)
                    img = Image(chart_path, width=220, height=160)
                    img.hAlign = 'CENTER'

                    combined = Table(
                        [[dist_table, img]],
                        colWidths=[CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.5]
                    )
                    combined.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
                    story.append(combined)
                except Exception as e:
                    print(f"Error creating sentiment chart: {e}")
                    story.append(dist_table)
                
                story.append(Spacer(1, 15))

            # Emotional tones
            tones = safe_list_value(sentiment.get("emotional_tones", []))
            if tones:
                tones_str = ", ".join([safe_string_value(t, "") for t in tones if safe_string_value(t, "")])
                if tones_str:
                    story.append(auto_paragraph(f"Emotional Tones: {tones_str}", styles, base_style='CustomBody'))
                    story.append(Spacer(1, 15))

        # Emotions section with comprehensive error handling
        emotions = safe_get_nested(data, 'analysis_results', 'emotions', default={}) or safe_get_nested(data, 'emotions', default={})

        if has_content(emotions):
            story.append(section_heading("Emotions", styles))
            story.append(Spacer(1, 10))

            conf_score = safe_numeric_value(emotions.get('confidence_score'), None)
            if conf_score is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {conf_score:.2f}", styles, base_style='CustomBody'))

            primary_emotions = safe_list_value(emotions.get('primary_emotions', []))
            if primary_emotions:
                emotion_strs = [safe_string_value(e, "") for e in primary_emotions if safe_string_value(e, "")]
                if emotion_strs:
                    story.append(auto_paragraph(f"<b>Primary Emotions:</b> {', '.join(emotion_strs)}", styles, base_style='CustomBody'))

            story.append(Spacer(1, 8))

            # Emotion intensity with safe numeric handling
            emotion_intensity = emotions.get('emotion_intensity', {})
            if isinstance(emotion_intensity, dict) and emotion_intensity:
                table_data = [[
                    auto_paragraph("Emotion", styles, bold=True),
                    auto_paragraph("Intensity", styles, bold=True),
                ]]
                
                for emo, val in emotion_intensity.items():
                    emo_str = safe_string_value(emo, "").capitalize()
                    val_num = safe_numeric_value(val, 0)
                    if emo_str:
                        table_data.append([
                            auto_paragraph(emo_str, styles),
                            auto_paragraph(f"{val_num:.0f}%", styles)

                        ])

                if len(table_data) > 1:  # Only create table if we have data
                    intensity_table = Table(table_data, colWidths=[100, 80])
                    intensity_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
                        ('BACKGROUND', (0,1), (-1,-1), RL_PALETTE["neutral_bg"]),
                        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
                        ('LEFTPADDING',(0,0),(-1,-1),6),
                        ('RIGHTPADDING',(0,0),(-1,-1),6),
                        ('TOPPADDING',(0,0),(-1,-1),4),
                        ('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ]))

                    # Safe radar chart creation
                    try:
                        chart_path = os.path.join(tempfile.gettempdir(), "emotion_radar.png")
                        save_radar_chart(emotion_intensity, chart_path)
                        radar_img = Image(chart_path, width=CHART_W, height=CHART_H)
                        radar_img._restrictSize(CHART_W, CHART_H)

                        combined = Table(
                            [[intensity_table, radar_img]],
                            colWidths=[CONTENT_WIDTH*0.35, CONTENT_WIDTH*0.65], 
                            rowHeights=[CHART_H + 20]
                        )

                        combined.setStyle(TableStyle([
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('ALIGN', (1,0), (1,0), 'CENTER'),
                        ]))

                        story.append(combined)
                    except Exception as e:
                        print(f"Error creating emotion radar chart: {e}")
                        story.append(intensity_table)
                    
                    story.append(Spacer(1, 15))

            # Emotional timeline
            timeline = safe_string_value(emotions.get('emotional_timeline', ""))
            if timeline:
                story.append(auto_paragraph("Emotional Timeline", styles, base_style='SubHeading'))
                story.append(auto_paragraph(timeline, styles, base_style='CustomBody'))
                story.append(Spacer(1, 10))

            # Speaker emotions
            speaker_emotions = emotions.get('speaker_emotions', {})
            if isinstance(speaker_emotions, dict) and speaker_emotions:
                story.append(auto_paragraph("Speaker Emotions", styles, base_style='SubHeading'))
                table_data = [[
                    auto_paragraph("Speaker", styles, bold=True),
                    auto_paragraph("Emotions", styles, bold=True),
                ]]
                
                for speaker, emos in speaker_emotions.items():
                    speaker_str = safe_string_value(speaker, "")
                    emos_list = safe_list_value(emos, [])
                    emos_str = ", ".join([safe_string_value(e, "") for e in emos_list if safe_string_value(e, "")])
                    
                    if speaker_str and emos_str:
                        table_data.append([
                            auto_paragraph(speaker_str, styles),
                            auto_paragraph(emos_str, styles)
                        ])

                if len(table_data) > 1:
                    spk_table = Table(table_data, colWidths=[180, CONTENT_WIDTH-190])
                    spk_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('ALIGN',(0,0),(-1,-1),'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
                        ('BACKGROUND', (0,1), (-1,-1), RL_PALETTE["neutral_bg"]),
                        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
                        ('LEFTPADDING',(0,0),(-1,-1),6),
                        ('RIGHTPADDING',(0,0),(-1,-1),6),
                        ('TOPPADDING',(0,0),(-1,-1),4),
                        ('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ]))
                    story.append(spk_table)
                story.append(Spacer(1, 20))

        # Action Items section with comprehensive error handling
        actions = safe_get_nested(data, 'analysis_results', 'action_items', default={}) or safe_get_nested(data, 'action_items', default={})

        if has_content(actions):
            story.append(section_heading("Action Items", styles))
            story.append(Spacer(1, 10))

            conf_score = safe_numeric_value(actions.get('confidence_score'), None)
            if conf_score is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {conf_score:.2f}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 6))

            action_list = safe_list_value(actions.get('action_items', []))
            if action_list:
                story.append(auto_paragraph("Identified Action Items", styles, base_style='SubHeading'))
                table_data = [[
                    auto_paragraph("Task", styles, bold=True),
                    auto_paragraph("Assignee", styles, bold=True),
                    auto_paragraph("Deadline", styles, bold=True),
                    auto_paragraph("Priority", styles, bold=True),
                ]]

                for item in action_list:
                    if isinstance(item, dict):
                        table_data.append([
                            auto_paragraph(safe_string_value(item.get("task"), "-"), styles),
                            auto_paragraph(safe_string_value(item.get("assignee"), "-"), styles),
                            auto_paragraph(safe_string_value(item.get("deadline"), "-"), styles),
                            auto_paragraph(safe_string_value(item.get("priority"), "-").capitalize(), styles),
                        ])

                if len(table_data) > 1:
                    act_table = Table(table_data, colWidths=[CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.15, CONTENT_WIDTH*0.2, CONTENT_WIDTH*0.15])
                    act_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT), 
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('ALIGN',(0,0),(-1,-1),'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),   
                        ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, HexColor("#CCCCCC")),
                        ('BOX', (0,0), (-1,-1), 0.5, HexColor("#CCCCCC")),
                        ('BACKGROUND', (0,1), (-1,-1), LIGHT_BG),    
                        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
                        ('LEFTPADDING',(0,0),(-1,-1),6),
                        ('RIGHTPADDING',(0,0),(-1,-1),6),
                        ('TOPPADDING',(0,0),(-1,-1),4),
                        ('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ]))
                    story.append(act_table)
                story.append(Spacer(1, 10))

            # Decisions made
            decisions = safe_list_value(actions.get('decisions_made', []))
            if decisions:
                story.append(auto_paragraph("Decisions Made", styles, base_style='SubHeading'))
                for d in decisions:
                    d_str = safe_string_value(d, "")
                    if d_str:
                        story.append(auto_paragraph(f"• {d_str}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 10))

            # Follow ups
            followups = safe_list_value(actions.get('follow_ups', []))
            if followups:
                story.append(auto_paragraph("Follow Ups", styles, base_style='SubHeading'))
                for f in followups:
                    f_str = safe_string_value(f, "")
                    if f_str:
                        story.append(auto_paragraph(f"• {f_str}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 20))

        # Topics section with comprehensive error handling
        topics_data = safe_get_nested(data, 'analysis_results', 'topics', default={}) or safe_get_nested(data, 'topics', default={})

        if has_content(topics_data):
            story.append(section_heading("Topics", styles))
            story.append(Spacer(1, 10))

            topics_conf = safe_numeric_value(topics_data.get("confidence_score"), None)
            if topics_conf is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {topics_conf:.2f}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 6))

            categories = safe_list_value(topics_data.get("topic_categories", []))
            if categories:
                story.append(auto_paragraph("Topic Categories:", styles, base_style='SubHeading'))
                cat_strs = [safe_string_value(c, "") for c in categories if safe_string_value(c, "")]
                if cat_strs:
                    cat_str = ", ".join(cat_strs)
                    story.append(auto_paragraph(cat_str, styles, base_style='CustomBody'))
                story.append(Spacer(1, 8))

            key_topics = safe_list_value(topics_data.get("key_topics", []))
            if key_topics:
                table_data = [[
                    auto_paragraph("Topic", styles, bold=True), 
                    auto_paragraph("Importance Score", styles, bold=True), 
                    auto_paragraph("Mentions", styles, bold=True), 
                    auto_paragraph("Context", styles, bold=True)
                ]]
                
                for kt in key_topics:
                    if isinstance(kt, dict):
                        table_data.append([
                            auto_paragraph(safe_string_value(kt.get("topic"), ""), styles),
                            auto_paragraph(f"{safe_numeric_value(kt.get('importance_score'), 0):.0f}%", styles),
                            auto_paragraph(str(safe_numeric_value(kt.get("mentions"), 0)), styles),
                            auto_paragraph(safe_string_value(kt.get("context"), ""), styles),
                        ])

                if len(table_data) > 1:
                    topics_table = Table(
                        table_data,
                        colWidths=[CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.18, CONTENT_WIDTH*0.12, CONTENT_WIDTH*0.45]
                    )
                    topics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                        ('BOX', (0,0), (-1,-1), 0.25, BORDER),
                        ('LEFTPADDING',(0,0),(-1,-1),6),
                        ('RIGHTPADDING',(0,0),(-1,-1),6),
                        ('TOPPADDING',(0,0),(-1,-1),4),
                        ('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ]))

                    story.append(topics_table)
                    story.append(Spacer(1, 12))

                    # Topics bar chart
                    try:
                        chart_path = os.path.join(tempfile.gettempdir(), "topics_bar.png")
                        save_bar_chart(key_topics, chart_path)
                        bar_img = Image(chart_path, width=CONTENT_WIDTH, height=220)  
                        story.append(bar_img)
                    except Exception as e:
                        print(f"Error creating topics bar chart: {e}")
                    
                    story.append(Spacer(1, 20))

        # Segment Sentiment section with comprehensive error handling
        segment_data = safe_get_nested(data, 'analysis_results', 'segment_sentiment', default={}) or safe_get_nested(data, 'segment_sentiment', default={})

        if has_content(segment_data):
            story.append(section_heading("Segment Sentiment", styles))
            story.append(Spacer(1, 10))

            seg_conf = safe_numeric_value(segment_data.get("confidence_score"), None)
            if seg_conf is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {seg_conf:.2f}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 6))

            timeline_summary = safe_string_value(segment_data.get("sentiment_timeline", ""))
            segments = safe_list_value(segment_data.get("segments", []))

            if timeline_summary or segments:
                if timeline_summary:
                    timeline_para = auto_paragraph(timeline_summary, styles, base_style='CustomBody')
                    story.append(timeline_para)
                    story.append(Spacer(1, 10))

                if segments:
                    table_data = [[
                        auto_paragraph("Start", styles, bold=True),
                        auto_paragraph("End", styles, bold=True),
                        auto_paragraph("Sentiment", styles, bold=True),
                        auto_paragraph("Score", styles, bold=True),
                        auto_paragraph("Content Summary", styles, bold=True),
                    ]]

                    for seg in segments:
                        if isinstance(seg, dict):
                            table_data.append([
                                auto_paragraph(safe_string_value(seg.get("start_time"), ""), styles),
                                auto_paragraph(safe_string_value(seg.get("end_time"), ""), styles),
                                auto_paragraph(safe_string_value(seg.get("sentiment"), "").title(), styles),
                                auto_paragraph(f"{safe_numeric_value(seg.get('sentiment_score'), 0):.0f}%", styles),

                                auto_paragraph(safe_string_value(seg.get("content_summary"), ""), styles),
                            ])

                    if len(table_data) > 1:
                        col_widths = [50, 50, 70, 50, CONTENT_WIDTH - 220]
                        seg_table = Table(table_data, colWidths=col_widths)
                        seg_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), ACCENT),
                            ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                            ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                            ('FONTSIZE', (0,0), (-1,-1), 9),
                            ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                            ('BOX', (0,0), (-1,-1), 0.25, BORDER),
                            ('LEFTPADDING',(0,0),(-1,-1),4),
                            ('RIGHTPADDING',(0,0),(-1,-1),4),
                            ('TOPPADDING',(0,0),(-1,-1),2),
                            ('BOTTOMPADDING',(0,0),(-1,-1),2),
                        ]))
                        story.append(seg_table)

                story.append(Spacer(1, 20))

        # Transcript section with comprehensive error handling
        transcript_data = safe_get_nested(data, 'analysis_results', 'transcript', default={}) or safe_get_nested(data, 'transcript', default={})

        if has_content(transcript_data):
            story.append(section_heading("Transcript", styles))
            story.append(Spacer(1, 10))

            trans_conf = safe_numeric_value(transcript_data.get("confidence_score"), None)
            if trans_conf is not None:
                story.append(auto_paragraph(f"<b>Confidence Score:</b> {trans_conf:.2f}", styles, base_style='CustomBody'))
                story.append(Spacer(1, 6))

            segments = safe_list_value(transcript_data.get("segments", []))
            if segments:
                table_data = [[
                    auto_paragraph("Speaker", styles, bold=True),
                    auto_paragraph("Text", styles, bold=True),
                    auto_paragraph("Start", styles, bold=True),
                    auto_paragraph("Lang", styles, bold=True),
                ]]

                for seg in segments:
                    if isinstance(seg, dict):
                        speaker = safe_string_value(seg.get("speaker_name") or seg.get("speaker"), "-")
                        text = safe_string_value(seg.get("text"), "-")
                        start_time = safe_string_value(seg.get("start_time"), "-")
                        language = safe_string_value(seg.get("language"), "-")
                        
                        table_data.append([
                            auto_paragraph(speaker, styles),
                            auto_paragraph(text, styles),
                            auto_paragraph(start_time, styles),
                            auto_paragraph(language, styles),
                        ])

                if len(table_data) > 1:
                    col_widths = [70, CONTENT_WIDTH - 240, 70, 70]
                    transcript_table = Table(table_data, colWidths=col_widths)

                    transcript_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), ACCENT),
                        ('TEXTCOLOR', (0,0), (-1,0), HexColor("#FFFFFF")),
                        ('FONTNAME', (0,0), (-1,0), 'noto_bold'),
                        ('FONTSIZE', (0,0), (-1,0), 11),
                        ('ALIGN', (0,0), (-1,0), 'CENTER'),
                        ('FONTNAME', (0,1), (-1,-1), 'noto_regular'),
                        ('FONTSIZE', (0,1), (-1,-1), 9),
                        ('VALIGN', (0,1), (-1,-1), 'TOP'),
                        ('ALIGN', (0,1), (-1,-1), 'LEFT'),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor("#FFFFFF"), LIGHT_BG]),
                        ('INNERGRID', (0,0), (-1,-1), 0.25, BORDER),
                        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('RIGHTPADDING', (0,0), (-1,-1), 6),
                        ('TOPPADDING', (0,0), (-1,-1), 4),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ]))

                    story.append(transcript_table)
                    story.append(Spacer(1, 18))
                    
                    # Transcript timeline chart
                    has_sentiment = any(isinstance(seg, dict) and "sentiment" in seg and "sentiment_score" in seg for seg in segments)
                    if has_sentiment:
                        try:
                            chart_path = os.path.join(tempfile.gettempdir(), "transcript_timeline.png")
                            save_transcript_timeline(segments, chart_path)
                            story.append(Spacer(1, 12))
                            story.append(Image(chart_path, width=400, height=180))
                            story.append(Spacer(1, 18))
                        except Exception as e:
                            print(f"Error creating transcript timeline chart: {e}")

        # Build the final PDF
        try:
            doc.build(story)
            return out_pdf_path
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise

    except Exception as e:
        print(f"Critical error in PDF creation: {e}")
        # Create minimal fallback PDF
        try:
            doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
            story = [
                Paragraph("Audio Analysis Report", getSampleStyleSheet()['Title']),
                Spacer(1, 20),
                Paragraph("Error occurred while generating full report.", getSampleStyleSheet()['Normal']),
                Paragraph(f"Error details: {str(e)}", getSampleStyleSheet()['Normal'])
            ]
            doc.build(story)
            return out_pdf_path
        except Exception as fallback_error:
            print(f"Even fallback PDF creation failed: {fallback_error}")
            raise

def generate_and_upload_report(data):
    """
    Enhanced report generation with comprehensive error handling.
    """
    api_url = "https://api.convlyze.com/api/users/upload"
    
    try:
        # Validate input
        if data is None:
            print("Warning: No data provided for report generation")
            data = {}

        if not isinstance(data, dict):
            print(f"Warning: Input data is not a dict (type: {type(data)}), converting...")
            data = {}

        # Validate data structure
        data = validate_data_structure(data)

        out_pdf = os.path.join(tempfile.gettempdir(), "convlyze_audio_analysis_report.pdf")
        
        # Create PDF with error handling
        create_pdf_from_json(data, out_pdf)
        
        # Verify PDF was created
        if not os.path.exists(out_pdf):
            raise Exception("PDF file was not created successfully")

        # Upload with error handling
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
        print(f"Error in generate_and_upload_report: {e}")
        raise

def create_fallback_report(error_message="Unknown error occurred"):
    """
    Create a basic fallback report when main report generation fails.
    """
    fallback_data = {
        'metadata': {
            'file_name': 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': 0,
            'primary_language': 'Unknown',
            'analysis_version': '1.0'
        },
        'analysis_results': {
            'summary': {
                'summary': f'Report generation failed: {error_message}',
                'confidence_score': 0.0
            }
        }
    }
    return fallback_data

# Export the main functions
__all__ = [
    'create_pdf_from_json',
    'generate_and_upload_report', 
    'validate_data_structure',
    'create_fallback_report',
    'safe_get_nested',
    'safe_numeric_value',
    'safe_list_value', 
    'safe_string_value'
]


if __name__ == "__main__":
    """Test report1.py locally with demo data"""
    import tempfile, os

    sample_data = {
        "validation": {
            "is_valid_interview": True,
            "content_type": "VALID_INTERVIEW",
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
            "multilingual_content": "single_language"
        },
        "analysis_results": {
            "summary": {
                "summary": "Strong technical candidate with excellent communication skills.",
                "key_points": ["Good Python knowledge", "Strong system design skills"],
                "main_topics": ["Python", "System Design", "Team Collaboration"]
            },
            "transcript": {
                "transcript": "Interviewer: Tell me about yourself.\nCandidate: I have 5 years of experience in backend development...",
                "segments": []
            },
            "candidate_assessment": {
                "overall_recommendation": {
                    "score": 85,
                    "recommendation": "HIRE",
                    "confidence_level": "HIGH",
                    "key_strengths": ["Technical depth", "Communication"],
                    "key_concerns": [],
                    "red_flags": [],
                    "next_steps": ["Schedule final round"]
                }
            }
        }
    }

    out_path = os.path.join(tempfile.gettempdir(), "test_report1_logo.pdf")
    print(f"Generating test PDF at: {out_path}")
    create_pdf_from_json(sample_data, out_path)
    print(f"✓ Done! Open the PDF to verify the new logo.")