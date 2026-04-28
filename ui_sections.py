import base64
import html
import io
import shutil
import textwrap
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


_ONBOARDING_GENDER_AVATAR_WIDTH_PX = 288
WELCOME_VIDEO = Path(__file__).parent / "img" / "welcome_video.mp4"
WELCOME_FALLBACK_IMAGE = Path(
    "/Users/bztr1ng2l1ve/.cursor/projects/Users-bztr1ng2l1ve-Desktop-ml-diet-twin-app/assets/image-f734546c-2013-4351-aaf2-4302d0c6611d.png"
)

# (male glob, female glob) — newest matching pair wins; add rows when Cursor renames uploads.
_GENDER_UPLOAD_PATTERN_PAIRS = (
    ("*20260420191450_72_131*.png", "*20260420191452_73_131*.png"),
    ("*771d108b*.png", "*57d53876*.png"),
)

def apply_custom_theme_styles() -> None:
    st.markdown(
        """
        <style>
            /* Global text override: every letter renders black */
            html, body, .stApp, .stApp * {
                color: #000000 !important;
            }
            :root {
                --apple-bg-light: #e8f8ee;
                --apple-text: #1d1d1f;
                --mint-50: #f0fdf7;
                --mint-100: #dcfce7;
                --mint-200: #bbf7d0;
                --mint-500: #22c55e;
                --mint-600: #16a34a;
                --mint-700: #15803d;
                --mint-900: #14532d;
            }
            .stApp {
                background: linear-gradient(180deg, #f0fdf7 0%, #dff6ea 42%, #c8ebdb 100%);
                color: var(--apple-text);
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 16px;
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main {
                background: transparent !important;
            }
            header[data-testid="stHeader"] {
                display: none !important;
                height: 0 !important;
            }
            [data-testid="stToolbar"] {
                display: none !important;
            }
            .main .block-container {
                max-width: 1240px;
                padding-top: 0.7rem;
                padding-bottom: 2.4rem;
                margin-left: 0 !important;
                margin-right: auto !important;
                padding-left: 0.1rem;
                padding-right: 0.85rem;
            }
            [data-testid="stAppViewContainer"] > .main .block-container {
                width: auto !important;
                max-width: 1240px !important;
                margin-left: 0 !important;
                margin-right: auto !important;
                padding-left: 0.05rem !important;
                padding-right: 0.85rem !important;
            }
            [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            .main .block-container {
                width: 100% !important;
                max-width: none !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding-left: 0.2rem !important;
                padding-right: 0.8rem !important;
            }
            [data-testid="stSidebar"] {
                background: var(--apple-bg-light);
                border-right: none !important;
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
                padding: 0 !important;
            }
            h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
                color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
                letter-spacing: -0.2px;
            }
            .app-title-wrap {
                text-align: center;
                margin-bottom: 0.7rem;
            }
            .app-title {
                font-size: 2.6rem;
                font-weight: 700;
                letter-spacing: -0.3px;
                line-height: 1.08;
                margin-bottom: 0.2rem;
            }
            .app-subtitle {
                font-size: 0.95rem;
                line-height: 1.35;
                color: rgba(20, 83, 45, 0.72);
                text-align: center;
                margin-bottom: 0.35rem;
            }
            .hero-card {
                background: #f7fff9;
                border: 1px solid rgba(22, 163, 74, 0.2);
                border-radius: 16px;
                box-shadow: none;
                padding: 18px 20px;
                margin-bottom: 14px;
            }
            .hero-dark {
                background: linear-gradient(180deg, #ffffff 0%, #f3fff8 100%);
                color: #14532d;
                border-radius: 20px;
                border: 1px solid rgba(22, 163, 74, 0.2);
                border-top: 4px solid var(--mint-600);
                box-shadow: rgba(21, 128, 61, 0.1) 0px 8px 18px;
            }
            .hero-dark .hero-title,
            .hero-dark .hero-sub {
                color: #14532d;
            }
            .hero-title {
                font-size: 1.31rem;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 4px;
            }
            .hero-sub {
                color: rgba(0, 0, 0, 0.8);
                font-size: 1rem;
                line-height: 1.47;
                margin-bottom: 10px;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid rgba(22, 163, 74, 0.16);
                border-radius: 16px;
                box-shadow: rgba(21, 128, 61, 0.08) 0px 6px 12px;
                padding: 14px 16px;
                margin-bottom: 8px;
            }
            .metric-label {
                font-size: 0.9rem;
                color: rgba(21, 128, 61, 0.78);
                margin-bottom: 6px;
            }
            .metric-value {
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 600;
                color: #14532d;
            }
            .section-title {
                font-size: 1.42rem;
                font-weight: 700;
                color: #14532d;
                margin: 12px 0;
            }
            .result-shell {
                max-width: 100%;
                margin: 0;
                margin-left: 0 !important;
                width: 100% !important;
            }
            .result-card {
                background: #ffffff;
                border: 1px solid rgba(22, 163, 74, 0.2);
                border-radius: 28px;
                box-shadow: rgba(21, 128, 61, 0.12) 0px 14px 30px;
                padding: 1.1rem 1.2rem 1.7rem 1.2rem;
                margin-right: auto;
            }
            .result-card .hero-card {
                margin-top: 1.5rem;
                margin-bottom: 1.8rem;
            }
            .result-card .metric-card {
                margin-bottom: 1.35rem;
            }
            .result-top-progress {
                height: 14px;
                border-radius: 999px;
                background: #e5e7eb;
                overflow: hidden;
                margin: 0.3rem 0 1.7rem 0;
            }
            .result-top-progress-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
            }
            .result-ready-title {
                color: #14532d;
                font-size: 2.15rem;
                font-weight: 900;
                letter-spacing: -0.3px;
                line-height: 1.04;
                margin: 0.1rem 0 0.15rem 0;
            }
            .result-ready-sub {
                color: rgba(20, 83, 45, 0.8);
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }
            .chip {
                display: inline-block;
                background: #ecfdf3;
                color: #166534;
                border: 1px solid rgba(34, 197, 94, 0.35);
                border-radius: 980px;
                padding: 2px 10px;
                margin: 3px 6px 0 0;
                font-size: 0.88rem;
            }
            .hero-dark .chip {
                background: #e8f9ef;
                color: #166534;
                border: 1px solid rgba(34, 197, 94, 0.35);
            }
            .small-note {
                color: rgba(20, 83, 45, 0.85);
                font-size: 0.95rem;
                line-height: 1.5;
            }
            .table-shell {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid rgba(22, 163, 74, 0.16);
                box-shadow: rgba(21, 128, 61, 0.08) 0px 4px 14px;
                padding: 8px 10px;
                margin-top: 6px;
                margin-bottom: 8px;
            }
            .mint-table-wrap {
                width: 100%;
                overflow-x: auto;
                border-radius: 12px;
                border: 1px solid #bbf7d0;
                background: #ffffff;
            }
            .mint-table {
                width: 100%;
                border-collapse: collapse;
                background: #ffffff;
                color: #111827 !important;
                font-size: 0.95rem;
            }
            .mint-table th {
                background: #ecfdf3;
                color: #14532d !important;
                text-align: left;
                font-weight: 700;
                padding: 9px 11px;
                border: 1px solid #bbf7d0;
                white-space: nowrap;
            }
            .mint-table td {
                background: #ffffff;
                color: #111827 !important;
                padding: 8px 11px;
                border: 1px solid #dcfce7;
                vertical-align: top;
            }
            .block-card {
                background: #f4fff7;
                border: 1px solid rgba(22, 163, 74, 0.14);
                border-radius: 16px;
                padding: 14px 14px 10px 14px;
                margin-bottom: 12px;
            }
            .twin-section-box {
                background: #ffffff;
                border: 1px solid rgba(22, 163, 74, 0.2);
                border-radius: 16px;
                box-shadow: rgba(21, 128, 61, 0.08) 0px 4px 12px;
                padding: 12px 14px 12px 14px;
                margin: 0.35rem 0 0.8rem 0;
            }
            .twin-section-box .section-title {
                margin-top: 0.05rem;
                margin-bottom: 0.45rem;
            }
            div[data-testid="stTabs"] {
                background: #ffffff;
                border: 1px solid rgba(22, 163, 74, 0.16);
                border-radius: 16px;
                padding: 14px 14px 14px 14px;
                box-shadow: rgba(21, 128, 61, 0.08) 0px 4px 12px;
                margin-top: 1.4rem;
            }
            [data-baseweb="tab-list"] {
                gap: 12px;
                margin-bottom: 0.55rem;
            }
            [data-baseweb="tab"] {
                border-radius: 999px !important;
                background: #ffffff !important;
                color: #166534 !important;
                border: 1px solid rgba(22, 163, 74, 0.4) !important;
                padding: 10px 22px !important;
                font-size: 1.03rem !important;
                font-weight: 800 !important;
                text-decoration: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            [aria-selected="true"][data-baseweb="tab"] {
                background: linear-gradient(180deg, #22c55e 0%, #16a34a 100%) !important;
                color: #ffffff !important;
                border-color: #16a34a !important;
            }
            [data-baseweb="tab-highlight"] {
                display: none !important;
                height: 0 !important;
                background: transparent !important;
            }
            [data-baseweb="tab"]::before,
            [data-baseweb="tab"]::after {
                content: none !important;
                display: none !important;
            }
            .stButton > button,
            .stFormSubmitButton > button {
                border-radius: 10px !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                padding: 0.52rem 1rem !important;
            }
            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                background: linear-gradient(180deg, #22c55e 0%, #16a34a 100%) !important;
                border: none !important;
                color: #ffffff !important;
            }
            .stButton > button[kind="secondary"],
            .stFormSubmitButton > button[kind="secondary"] {
                background: #ffffff !important;
                color: #166534 !important;
                border: 1px solid rgba(22, 163, 74, 0.3) !important;
            }
            [data-testid="stProgress"] > div > div > div > div {
                background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%) !important;
            }
            .stProgress > div {
                background: #ffffff !important;
                border: 1px solid rgba(22, 163, 74, 0.16) !important;
                border-radius: 12px !important;
                padding: 3px !important;
            }
            .result-shell .stProgress {
                margin-bottom: 2rem;
                margin-top: 1rem;
            }
            body:has(.result-shell) .result-card [data-testid="stHorizontalBlock"] {
                gap: 1.4rem !important;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid rgba(22, 163, 74, 0.2);
                border-radius: 10px;
            }
            [data-testid="stDataFrame"] * {
                color: #14532d !important;
            }
            [data-testid="stTable"] {
                border: 1px solid rgba(22, 163, 74, 0.2) !important;
                border-radius: 12px !important;
                overflow: hidden !important;
                background: #ffffff !important;
            }
            [data-testid="stTable"] table {
                width: 100% !important;
                border-collapse: collapse !important;
                color: #111827 !important;
                background: #ffffff !important;
            }
            [data-testid="stTable"] th {
                background: #ecfdf3 !important;
                color: #14532d !important;
                font-weight: 700 !important;
                border-bottom: 1px solid #bbf7d0 !important;
                padding: 8px 10px !important;
                text-align: left !important;
            }
            [data-testid="stTable"] td {
                background: #ffffff !important;
                color: #111827 !important;
                border-top: 1px solid #dcfce7 !important;
                padding: 8px 10px !important;
            }
            [data-testid="stAlert"] {
                border-radius: 10px !important;
                border: 1px solid rgba(22, 163, 74, 0.22) !important;
                background: #f0fdf7 !important;
                color: #14532d !important;
            }
            [data-testid="stExpander"] {
                background: #ffffff;
                border: 1px solid rgba(22, 163, 74, 0.16);
                border-radius: 14px;
                box-shadow: rgba(21, 128, 61, 0.06) 0px 4px 10px;
            }
            [data-testid="stExpander"] details {
                background: #ffffff !important;
                border-radius: 14px !important;
            }
            [data-testid="stExpander"] summary {
                background: #ecfdf3 !important;
                border: 1px solid #bbf7d0 !important;
                border-radius: 12px !important;
                color: #14532d !important;
            }
            [data-testid="stExpander"] summary * {
                color: #14532d !important;
            }
            [data-testid="stExpander"] summary:hover {
                background: #dcfce7 !important;
            }
            /* Plan screen: force all text to black */
            body:has(.result-shell) .result-shell,
            body:has(.result-shell) .result-shell * {
                color: #111827 !important;
            }
            .welcome-bleed {
                position: fixed;
                inset: 0;
                width: 100vw;
                height: 100svh;
                margin: 0 !important;
                z-index: 1;
            }
            .welcome-bleed,
            .welcome-bleed * {
                color: #ffffff !important;
            }
            .welcome-hero {
                width: 100%;
                height: 100%;
                min-height: 100%;
                border-radius: 0;
                position: relative;
                overflow: hidden;
                margin-bottom: 0;
                box-shadow: none;
            }
            .welcome-video {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                z-index: 0;
            }
            .welcome-hero::before {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(90deg, rgba(0, 0, 0, 0.52) 0%, rgba(0, 0, 0, 0.28) 50%, rgba(0, 0, 0, 0.58) 100%);
            }
            .welcome-inner {
                position: relative;
                z-index: 2;
                height: 100%;
                color: #ffffff;
                padding: 24px 28px;
                display: flex;
                flex-direction: column;
            }
            .welcome-top {
                display: flex;
                justify-content: flex-start;
                align-items: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.25);
                padding-bottom: 10px;
            }
            .welcome-brand {
                font-size: 2.05rem;
                font-weight: 700;
                color: #ffffff;
                text-align: left;
            }
            .welcome-main {
                display: grid;
                grid-template-columns: 2.1fr 1fr;
                gap: 24px;
                flex: 1;
                align-items: stretch;
                margin-top: 42px;
            }
            .welcome-kicker {
                font-size: 1.32rem;
                font-weight: 600;
                margin-top: 28px;
                margin-left: 16px;
                color: #ffffff;
            }
            .welcome-title {
                margin-top: 10px;
                font-size: 5rem;
                line-height: 1.02;
                font-weight: 800;
                letter-spacing: -0.6px;
                color: #ffffff;
                margin-left: 16px;
            }
            .welcome-stats {
                border-left: 1px solid rgba(255, 255, 255, 0.26);
                padding-left: 26px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 100%;
            }
            .welcome-stat {
                padding: 28px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.22);
                min-height: 22%;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .welcome-stat-value {
                font-size: 2.05rem;
                font-weight: 700;
                line-height: 1.15;
                color: #ffffff;
            }
            .welcome-stat-label {
                margin-top: 10px;
                font-size: 1.22rem;
                line-height: 1.45;
                color: rgba(255, 255, 255, 0.86);
            }
            .welcome-inline-cta {
                position: absolute;
                left: 36px;
                top: 49%;
                transform: none;
                z-index: 8;
                min-width: 280px;
                text-align: center;
                background: rgba(255, 255, 255, 0.96);
                color: #111111 !important;
                border: 1px solid rgba(255, 255, 255, 1);
                border-radius: 4px;
                font-weight: 700;
                font-size: 1.18rem;
                letter-spacing: 0.4px;
                padding: 16px 26px;
                text-decoration: none !important;
                box-shadow: rgba(0, 0, 0, 0.25) 0px 8px 20px;
            }
            .welcome-inline-cta:hover {
                background: #ffffff;
                color: #111111 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_hero(video_uri: str, fallback_style: str) -> None:
    video_html = (
        f'<video class="welcome-video" autoplay muted loop playsinline preload="auto"><source src="{video_uri}" type="video/mp4"></video>'
        if video_uri
        else ""
    )
    st.html(
        textwrap.dedent(
            f"""
            <div class="welcome-bleed">
                <div class="welcome-hero" style="{fallback_style}">
                    {video_html}
                    <div class="welcome-inner">
                        <div class="welcome-top">
                            <div class="welcome-brand">Diet Twin Planner</div>
                        </div>
                        <div class="welcome-main">
                            <div>
                                <div class="welcome-title">Achieve Your<br/>Fitness Goals</div>
                                <div class="welcome-kicker">Customized for real results</div>
                            </div>
                            <div class="welcome-stats">
                                <div class="welcome-stat">
                                    <div class="welcome-stat-value">Profile-Based Planning</div>
                                    <div class="welcome-stat-label">Recommendations tailored to your age, body data, and goal weight</div>
                                </div>
                                <div class="welcome-stat">
                                    <div class="welcome-stat-value">Meal Structure Guidance</div>
                                    <div class="welcome-stat-label">Clear daily meal direction you can use without overcomplication</div>
                                </div>
                                <div class="welcome-stat">
                                    <div class="welcome-stat-value">Diet Twin Matching</div>
                                    <div class="welcome-stat-label">Find your closest body profile using your age, height, weight, and goals</div>
                                </div>
                                <div class="welcome-stat">
                                    <div class="welcome-stat-value">Personalized Lift Plan</div>
                                    <div class="welcome-stat-label">Get a practical weekly workout split generated from your profile</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <a class="welcome-inline-cta" href="?welcome_start=1">GET STARTED</a>
                </div>
            </div>
            """
        )
    )


def render_welcome_page_layout(video_uri: str, image_uri: str) -> None:
    st.html(
        textwrap.dedent(
            """
            <style>
                html, body, [data-testid="stAppViewContainer"] {
                    margin: 0 !important;
                    padding: 0 !important;
                    background: #111111 !important;
                    overflow: hidden !important;
                }
                [data-testid="stAppViewContainer"] > .main {
                    padding-top: 0 !important;
                }
                .main .block-container {
                    max-width: 100% !important;
                    padding-top: 0 !important;
                    padding-bottom: 0 !important;
                    padding-left: 0 !important;
                    padding-right: 0 !important;
                }
                .stApp {
                    background: #111111 !important;
                    overflow-x: hidden !important;
                }
                .welcome-bleed {
                    margin-top: 0 !important;
                }
            </style>
            """
        )
    )
    if image_uri:
        fallback_style = f"background-image: url('{image_uri}'); background-size: cover; background-position: center center;"
    else:
        fallback_style = "background: linear-gradient(120deg, #1f2937, #0f172a);"
    render_welcome_hero(video_uri, fallback_style)


def render_app_title() -> None:
    st.markdown(
        """
        <div class="app-title-wrap">
            <div class="app-title">Diet Twin Planner</div>
            <div class="app-subtitle">Premium, minimal planning interface with focused nutrition and training decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data
def get_welcome_bg_data_uri():
    candidates = [
        Path(__file__).parent / "assets" / "welcome_page.png",
        WELCOME_FALLBACK_IMAGE,
    ]
    for image_path in candidates:
        if image_path.exists():
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    return ""


@st.cache_data
def get_welcome_video_data_uri():
    if not WELCOME_VIDEO.exists():
        return ""
    encoded = base64.b64encode(WELCOME_VIDEO.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


# (male glob, female glob) — newest matching pair wins; add rows when Cursor renames uploads.
_GENDER_UPLOAD_PATTERN_PAIRS = (
    ("*20260420191450_72_131*.png", "*20260420191452_73_131*.png"),
    ("*771d108b*.png", "*57d53876*.png"),
)


def _latest_png_match(directory: Path, pattern: str) -> Path | None:
    hits = [p for p in directory.glob(pattern) if p.is_file()]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _pick_gender_sources_from_dir(src_dir: Path) -> tuple[Path | None, Path | None]:
    for pat_m, pat_f in _GENDER_UPLOAD_PATTERN_PAIRS:
        m = _latest_png_match(src_dir, pat_m)
        f = _latest_png_match(src_dir, pat_f)
        if m and f:
            return m, f
    return None, None


def _gender_avatar_cursor_assets_dir() -> Path | None:
    """Cursor stores pasted chat images under ~/.cursor/projects/<id>/assets/."""
    root = Path.home() / ".cursor" / "projects"
    if not root.is_dir():
        return None
    explicit = root / "c-Users-Administrator-Desktop-Diet-Exercise-Plan-App" / "assets"
    if explicit.is_dir():
        return explicit
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        ad = proj / "assets"
        if not ad.is_dir():
            continue
        if _pick_gender_sources_from_dir(ad)[0]:
            return ad
    return None


def sync_gender_avatars_to_assets() -> None:
    """Copy chat-upload avatars into ./assets when missing (UUID suffix may change)."""
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest_m = assets_dir / "gender_avatar_male.png"
    dest_f = assets_dir / "gender_avatar_female.png"
    if dest_m.is_file() and dest_f.is_file():
        return
    src_dir = _gender_avatar_cursor_assets_dir()
    if not src_dir:
        return
    src_m, src_f = _pick_gender_sources_from_dir(src_dir)
    try:
        if src_m and not dest_m.is_file():
            shutil.copy2(src_m, dest_m)
        if src_f and not dest_f.is_file():
            shutil.copy2(src_f, dest_f)
    except OSError:
        pass


def resolve_gender_avatar_paths():
    """Prefer ./assets; else newest matching files in Cursor upload folder."""
    sync_gender_avatars_to_assets()
    assets_dir = Path(__file__).parent / "assets"
    local_m = assets_dir / "gender_avatar_male.png"
    local_f = assets_dir / "gender_avatar_female.png"
    male = local_m if local_m.is_file() else None
    female = local_f if local_f.is_file() else None
    if male and female:
        return male, female
    src_dir = _gender_avatar_cursor_assets_dir()
    if not src_dir:
        return male, female
    src_m, src_f = _pick_gender_sources_from_dir(src_dir)
    if male is None:
        male = src_m
    if female is None:
        female = src_f
    return male, female


@st.cache_data(show_spinner=False)
def _gender_avatar_pil_transparent(path_str: str, mtime: float) -> Image.Image:
    """
    Remove outer near-white background by flood-fill from edges (keeps interior whites
    that are not connected to the border, e.g. clothing).
    """
    im = Image.open(Path(path_str)).convert("RGBA")
    arr = np.asarray(im, dtype=np.uint8).copy()
    h, w = arr.shape[0], arr.shape[1]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rgb_sum = r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)
    cand = (r >= 234) & (g >= 234) & (b >= 234) & (rgb_sum >= 705)
    visited = np.zeros((h, w), dtype=bool)
    dq: deque[tuple[int, int]] = deque()
    for j in range(w):
        for i in (0, h - 1):
            if cand[i, j] and not visited[i, j]:
                visited[i, j] = True
                dq.append((i, j))
    for i in range(h):
        for j in (0, w - 1):
            if cand[i, j] and not visited[i, j]:
                visited[i, j] = True
                dq.append((i, j))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and cand[ny, nx]:
                visited[ny, nx] = True
                dq.append((ny, nx))
    arr[:, :, 3] = np.where(visited, 0, arr[:, :, 3])
    return Image.fromarray(arr)


def _gender_pick_cell_html_from_pil(pil: Image.Image, label: str, width_px: int, sex_value: str) -> str:
    """Single HTML block: avatar + label (no Streamlit widget gap between them)."""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    esc = html.escape(label)
    title = html.escape(f"Select {label}")
    w = int(width_px)
    href = html.escape(f"?onboarding_sex={sex_value}")
    return (
        f'<a class="gender-pick-link" href="{href}" target="_self" rel="noopener" title="{title}" aria-label="{title}">'
        f'<div class="gender-pick-cell" role="presentation">'
        f'<img class="gender-inline-avatar" src="data:image/png;base64,{b64}" alt="" '
        f'width="{w}" style="width:{w}px;max-width:min({w}px,88vw);height:auto;display:block;margin:0;padding:0;border:0;" />'
        f'<div class="gender-inline-label">{esc}</div>'
        f"</div>"
        f"</a>"
    )


def _gender_pick_cell_html_emoji(emoji: str, label: str, _width_px: int, sex_value: str) -> str:
    esc = html.escape(label)
    title = html.escape(f"Select {label}")
    href = html.escape(f"?onboarding_sex={sex_value}")
    return (
        f'<a class="gender-pick-link" href="{href}" target="_self" rel="noopener" title="{title}" aria-label="{title}">'
        f'<div class="gender-pick-cell gender-pick-cell-emoji" role="presentation">'
        f'<div class="gender-emoji-fallback" aria-hidden="true">{emoji}</div>'
        f'<div class="gender-inline-label">{esc}</div>'
        f"</div>"
        f"</a>"
    )


def render_welcome_page():
    params = st.query_params
    clicked = params.get("welcome_start", "0") == "1"
    if clicked:
        params.clear()
        return True

    video_uri = get_welcome_video_data_uri()
    image_uri = get_welcome_bg_data_uri()
    render_welcome_page_layout(video_uri, image_uri)
    return False




def set_onboarding_sex(value: str) -> None:
    """Callback runs before the rest of the script; do not call st.rerun() here."""
    st.session_state["onboarding_selected_sex"] = value


def render_onboarding_wizard():
    defaults = {
        "name": "",
        "age": 24,
        "sex": "M",
        "height_cm": 172,
        "weight_kg": 75,
        "goal_weight_kg": 68,
        "days_per_week": 4,
        "schedule_type": "Regular daytime",
        "workout_location": "Home",
        "workout_time": "30-45 minutes",
        "diet_preference": "No preference",
        "craving_level": "Sometimes",
        "stress_level": "Medium",
        "sleep_quality": "Average",
        "health_conditions": ["None"],
    }
    activity_options = [
        "Sedentary",
        "Lightly Active",
        "Moderately Active",
        "Very Active",
    ]
    activity_to_days = {
        "Sedentary": 1,
        "Lightly Active": 3,
        "Moderately Active": 5,
        "Very Active": 6,
    }
    days_to_activity = {v: k for k, v in activity_to_days.items()}
    params = st.query_params

    def _qp_str(key: str, default: str = "") -> str:
        """Streamlit query params may return scalars or lists; normalize safely."""
        value = params.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return str(value).strip()

    param_sex = _qp_str("onboarding_sex").upper()
    param_height_raw = _qp_str("onboarding_height")
    param_weight_raw = _qp_str("onboarding_weight")
    param_goal_weight_raw = _qp_str("onboarding_goal_weight")
    param_birth_month_raw = _qp_str("onboarding_birth_month")
    param_birth_day_raw = _qp_str("onboarding_birth_day")
    param_birth_year_raw = _qp_str("onboarding_birth_year")
    param_goal_month_raw = _qp_str("onboarding_goal_month")
    param_goal_day_raw = _qp_str("onboarding_goal_day")
    param_goal_year_raw = _qp_str("onboarding_goal_year")
    param_next = _qp_str("onboarding_next", "0")
    param_age_next = _qp_str("onboarding_age_next", "0")
    param_weight_next = _qp_str("onboarding_weight_next", "0")
    param_goal_weight_next = _qp_str("onboarding_goal_weight_next", "0")
    param_back = _qp_str("onboarding_back", "0")
    param_weight_back = _qp_str("onboarding_weight_back", "0")
    param_age_back = _qp_str("onboarding_age_back", "0")
    param_goal_back = _qp_str("onboarding_goal_back", "0")
    param_goal_time_back = _qp_str("onboarding_goal_time_back", "0")
    param_plan_intro_back = _qp_str("onboarding_plan_intro_back", "0")
    param_details_back = _qp_str("onboarding_details_back", "0")
    param_welcome_back = _qp_str("onboarding_welcome_back", "0")
    if param_weight_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        st.session_state["onboarding_stage"] = "height"
        params.clear()
        st.rerun()
    if param_age_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_height_raw.isdigit():
            parsed_height = int(param_height_raw)
            if 145 <= parsed_height <= 230:
                st.session_state["onboarding_height_cm"] = parsed_height
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        st.session_state["onboarding_stage"] = "weight"
        params.clear()
        st.rerun()
    if param_goal_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_height_raw.isdigit():
            parsed_height = int(param_height_raw)
            if 145 <= parsed_height <= 230:
                st.session_state["onboarding_height_cm"] = parsed_height
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        if param_goal_weight_raw:
            try:
                parsed_goal_weight = float(param_goal_weight_raw)
            except ValueError:
                parsed_goal_weight = None
            if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
        st.session_state["onboarding_stage"] = "age"
        params.clear()
        st.rerun()
    if param_goal_time_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_height_raw.isdigit():
            parsed_height = int(param_height_raw)
            if 145 <= parsed_height <= 230:
                st.session_state["onboarding_height_cm"] = parsed_height
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        if param_goal_weight_raw:
            try:
                parsed_goal_weight = float(param_goal_weight_raw)
            except ValueError:
                parsed_goal_weight = None
            if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
        st.session_state["onboarding_stage"] = "goal_weight"
        params.clear()
        st.rerun()
    if param_plan_intro_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_height_raw.isdigit():
            parsed_height = int(param_height_raw)
            if 145 <= parsed_height <= 230:
                st.session_state["onboarding_height_cm"] = parsed_height
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        if param_goal_weight_raw:
            try:
                parsed_goal_weight = float(param_goal_weight_raw)
            except ValueError:
                parsed_goal_weight = None
            if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
        st.session_state["onboarding_stage"] = "goal_timeline"
        params.clear()
        st.rerun()
    if param_details_back == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_height_raw.isdigit():
            parsed_height = int(param_height_raw)
            if 145 <= parsed_height <= 230:
                st.session_state["onboarding_height_cm"] = parsed_height
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        if param_goal_weight_raw:
            try:
                parsed_goal_weight = float(param_goal_weight_raw)
            except ValueError:
                parsed_goal_weight = None
            if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
        st.session_state["onboarding_stage"] = "plan_intro"
        params.clear()
        st.rerun()
    if param_back == "1":
        st.session_state.pop("onboarding_selected_sex", None)
        st.session_state["onboarding_stage"] = "height"
        params.clear()
        st.rerun()
    if param_welcome_back == "1":
        st.session_state["welcome_seen"] = False
        params.clear()
        st.rerun()
    if param_height_raw.isdigit():
        parsed_height = int(param_height_raw)
        if 145 <= parsed_height <= 230:
            st.session_state["onboarding_height_cm"] = parsed_height
    if param_weight_raw:
        try:
            parsed_weight = float(param_weight_raw)
        except ValueError:
            parsed_weight = None
        if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
            current_weight_state = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
            if abs(current_weight_state - parsed_weight) >= 0.05:
                st.session_state["onboarding_weight_kg"] = parsed_weight
    if param_goal_weight_raw:
        try:
            parsed_goal_weight = float(param_goal_weight_raw)
        except ValueError:
            parsed_goal_weight = None
        if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
            current_goal_state = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
            if abs(current_goal_state - parsed_goal_weight) >= 0.05:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
    if param_birth_month_raw.isdigit():
        m = int(param_birth_month_raw)
        if 1 <= m <= 12:
            st.session_state["onboarding_birth_month"] = m
    if param_birth_day_raw.isdigit():
        d = int(param_birth_day_raw)
        if 1 <= d <= 31:
            st.session_state["onboarding_birth_day"] = d
    if param_birth_year_raw.isdigit():
        y = int(param_birth_year_raw)
        cy = date.today().year
        if cy - 75 <= y <= cy - 15:
            st.session_state["onboarding_birth_year"] = y
    if param_goal_month_raw.isdigit():
        m = int(param_goal_month_raw)
        if 1 <= m <= 12:
            st.session_state["onboarding_goal_month"] = m
    if param_goal_day_raw.isdigit():
        d = int(param_goal_day_raw)
        if 1 <= d <= 31:
            st.session_state["onboarding_goal_day"] = d
    if param_goal_year_raw.isdigit():
        y = int(param_goal_year_raw)
        cy = date.today().year
        if cy <= y <= cy + 5:
            st.session_state["onboarding_goal_year"] = y
    if param_age_next == "1":
        today = date.today()
        by = int(st.session_state.get("onboarding_birth_year", today.year - int(defaults["age"])))
        bm = int(st.session_state.get("onboarding_birth_month", 1))
        bd = int(st.session_state.get("onboarding_birth_day", 1))
        safe_bd = bd
        while safe_bd > 28:
            try:
                _ = date(by, bm, safe_bd)
                break
            except ValueError:
                safe_bd -= 1
        dob = date(by, bm, safe_bd)
        age_years = int((today - dob).days // 365.2425)
        age_years = max(16, min(75, age_years))
        st.session_state["onboarding_birth_day"] = safe_bd
        st.session_state["onboarding_age"] = age_years
        st.session_state["onboarding_stage"] = "goal_weight"
        params.clear()
        st.rerun()
    if param_weight_next == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        st.session_state["onboarding_stage"] = "age"
        params.clear()
        st.rerun()
    if param_goal_weight_next == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        if param_weight_raw:
            try:
                parsed_weight = float(param_weight_raw)
            except ValueError:
                parsed_weight = None
            if parsed_weight is not None and 35.0 <= parsed_weight <= 250.0:
                st.session_state["onboarding_weight_kg"] = parsed_weight
        if param_goal_weight_raw:
            try:
                parsed_goal_weight = float(param_goal_weight_raw)
            except ValueError:
                parsed_goal_weight = None
            if parsed_goal_weight is not None and 35.0 <= parsed_goal_weight <= 250.0:
                st.session_state["onboarding_goal_weight_kg"] = parsed_goal_weight
        st.session_state["onboarding_stage"] = "goal_timeline"
        params.clear()
        st.rerun()
    if param_sex in {"M", "F"}:
        st.session_state["onboarding_selected_sex"] = param_sex
        if param_next != "1" and param_weight_next != "1" and param_goal_weight_next != "1":
            params.clear()
            st.rerun()
    if param_next == "1":
        st.session_state["onboarding_selected_sex"] = (
            param_sex
            if param_sex in {"M", "F"}
            else st.session_state.get("onboarding_selected_sex", "M")
        )
        st.session_state["onboarding_stage"] = "weight"
        params.clear()
        st.rerun()
    selected_sex = st.session_state.get("onboarding_selected_sex")
    show_gender_page = selected_sex is None
    bridge_sex = selected_sex if selected_sex in {"M", "F"} else "M"
    bridge_weight = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
    bridge_height = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
    bridge_script_html = """
        <script>
            (function () {
                if (window.__onboardingBridgeInit) return;
                window.__onboardingBridgeInit = true;
                const defaultSex = "__BRIDGE_SEX__";
                const defaultWeight = "__BRIDGE_WEIGHT__";
                const defaultHeight = "__BRIDGE_HEIGHT__";

                function updateLink(id, href) {
                    const link = document.getElementById(id);
                    if (link) link.setAttribute("href", href);
                }

                window.addEventListener("message", function (event) {
                    const data = event && event.data ? event.data : {};
                    if (!data || typeof data !== "object") return;

                    if (data.type === "onboarding-height") {
                        const value = Number(data.value);
                        const sex = String(data.sex || defaultSex).toUpperCase() === "F" ? "F" : "M";
                        if (!Number.isFinite(value)) return;
                        updateLink("height-next-link", `?onboarding_sex=${sex}&onboarding_height=${Math.round(value)}&onboarding_next=1`);
                        return;
                    }

                    if (data.type === "onboarding-weight") {
                        const value = Number(data.value);
                        const sex = String(data.sex || defaultSex).toUpperCase() === "F" ? "F" : "M";
                        if (!Number.isFinite(value)) return;
                        updateLink("weight-next-link", `?onboarding_sex=${sex}&onboarding_height=${defaultHeight}&onboarding_weight=${value.toFixed(1)}&onboarding_weight_next=1`);
                        return;
                    }

                    if (data.type === "onboarding-goal-weight") {
                        const value = Number(data.value);
                        const sex = String(data.sex || defaultSex).toUpperCase() === "F" ? "F" : "M";
                        if (!Number.isFinite(value)) return;
                        updateLink("goal-weight-next-link", `?onboarding_sex=${sex}&onboarding_height=${defaultHeight}&onboarding_weight=${defaultWeight}&onboarding_goal_weight=${value.toFixed(1)}&onboarding_goal_weight_next=1`);
                        updateLink("goal-weight-back-link", `?onboarding_sex=${sex}&onboarding_height=${defaultHeight}&onboarding_weight=${defaultWeight}&onboarding_goal_weight=${value.toFixed(1)}&onboarding_goal_back=1`);
                    }
                });
            })();
        </script>
        """
    st.html(
        bridge_script_html
        .replace("__BRIDGE_SEX__", bridge_sex)
        .replace("__BRIDGE_WEIGHT__", f"{bridge_weight:.1f}")
        .replace("__BRIDGE_HEIGHT__", str(bridge_height)),
        unsafe_allow_javascript=True,
    )

    def bmi_status_label(bmi_value: float) -> str:
        if bmi_value < 18.5:
            return "Underweight"
        if bmi_value < 25:
            return "Ideal"
        if bmi_value < 30:
            return "Overweight"
        return "Obese"

    st.markdown(
        """
        <style>
            /* Onboarding: mint page background + white card (reference UI) */
            .stApp {
                background: linear-gradient(180deg, #f0fdf7 0%, #dff6ea 42%, #c8ebdb 100%) !important;
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main {
                background: transparent !important;
            }
            /* Tighten Streamlit default top padding on onboarding only */
            .main .block-container {
                padding-top: 0 !important;
                padding-bottom: 1.25rem !important;
                background: transparent !important;
            }
            [data-testid="stAppViewContainer"] > .main {
                padding-top: 0 !important;
            }
            .assessment-shell {
                max-width: 460px;
                margin: 0 auto;
                padding: 12px 20px 22px 20px;
            }
            .assessment-card {
                background: #ffffff;
                border-radius: 34px;
                border: none;
                box-shadow: 0 18px 50px rgba(20, 90, 70, 0.1);
                padding: 1.35rem 1.25rem 1.45rem 1.25rem;
            }
            body:has(.assessment-gender-step) .assessment-card.assessment-gender-step {
                margin-top: 0;
                padding: 1.05rem 1.25rem 1.25rem 1.25rem;
            }
            body:has(.assessment-gender-step) .assessment-progress {
                margin-bottom: 0.95rem;
            }
            body:has(.assessment-gender-step) .assessment-question {
                margin-top: 0;
                margin-bottom: 0.28rem;
            }
            body:has(.assessment-gender-step) .assessment-helper {
                margin-bottom: 0.55rem;
            }
            body:has(.assessment-gender-step) .gender-pick-gap {
                height: 0.5rem;
            }
            body:has(.assessment-gender-step) .main .block-container {
                margin-top: 0 !important;
            }
            /* Keep gender step centered in viewport (remove sidebar offset) */
            body:has(.assessment-gender-step) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            .assessment-progress {
                height: 10px;
                border-radius: 999px;
                background: #e8ecef;
                overflow: hidden;
                margin-bottom: 1.3rem;
                position: relative;
                box-sizing: border-box;
            }
            .assessment-progress-fill {
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: 12%;
                background: linear-gradient(90deg, #22c55e, #16a34a);
                border-radius: 999px;
            }
            body:has(.weight-stage-lock) .assessment-card .assessment-progress,
            body:has(.age-stage-lock) .assessment-card .assessment-progress,
            body:has(.goal-stage-lock) .assessment-card .assessment-progress,
            body:has(.goal-time-stage-lock) .assessment-card .assessment-progress,
            body:has(.plan-intro-stage-lock) .assessment-card .assessment-progress,
            body:has(.details-stage-lock) .assessment-card .assessment-progress {
                width: auto !important;
                margin-left: 1.25rem !important;
                margin-right: 1.25rem !important;
            }
            /* Avoid Streamlit markdown “white strip” sitting above the real track */
            .assessment-shell [data-testid="stMarkdownContainer"] {
                background: transparent !important;
            }
            .assessment-shell [data-testid="stMarkdownContainer"] > div {
                background: transparent !important;
            }
            .gender-pick-gap {
                height: 1.5rem;
            }
            .gender-pick-link {
                text-decoration: none !important;
                color: inherit !important;
                display: inline-flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center;
                width: 100%;
            }
            .gender-pick-link .gender-inline-label {
                margin: 0.08rem 0 0 0 !important;
                padding: 0 !important;
                font-size: 1.78rem !important;
                font-weight: 900 !important;
                color: #0f172a !important;
                text-align: center !important;
                line-height: 1.05 !important;
                letter-spacing: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
            }
            .assessment-shell [data-testid="column"] {
                background: transparent !important;
                text-align: center;
            }
            /* Gender row: 50/50; columns not inside .assessment-shell in DOM — scope by gender step card */
            body:has(.assessment-gender-step) [data-testid="stHorizontalBlock"] {
                width: 100%;
                display: flex !important;
                flex-direction: row !important;
                justify-content: stretch !important;
                align-items: flex-start !important;
                gap: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stHorizontalBlock"] [data-testid="column"] {
                flex: 1 1 0 !important;
                min-width: 0 !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }
            body:has(.assessment-gender-step) [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
                width: 100%;
                display: flex !important;
                justify-content: center !important;
                align-items: flex-start !important;
                flex: 0 0 auto !important;
                min-height: 0 !important;
                height: auto !important;
            }
            /* Nested [pad | mid | pad]: don’t stretch mid column to match side columns (was creating huge empty space above labels) */
            body:has(.assessment-gender-step) [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stVerticalBlock"] [data-testid="stHorizontalBlock"] {
                align-items: flex-start !important;
                gap: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stVerticalBlock"] {
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                flex: 0 0 auto !important;
                min-height: 0 !important;
                height: auto !important;
            }
            /* No hover "help" popover / element toolbar on gender pickers */
            .assessment-shell [data-testid="stElementToolbar"] {
                display: none !important;
            }
            /* Gender: avatar + label in one HTML block (no st.image / st.markdown gap) */
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])) [data-testid="element-container"]:has(.gender-pick-cell) {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                width: 100% !important;
                margin-top: 0 !important;
                margin-bottom: 0 !important;
                position: relative !important;
                z-index: 0 !important;
                /* Entire markdown block must ignore hits so the transparent st.button (above in z-order) receives the click */
                pointer-events: none !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])) [data-testid="element-container"]:has(.gender-pick-cell) * {
                pointer-events: none !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])) [data-testid="element-container"]:has(.gender-pick-cell) [data-testid="stMarkdownContainer"] {
                margin: 0 !important;
                padding: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0 !important;
                row-gap: 0 !important;
                width: 100%;
                max-width: 320px;
                margin: 0 auto;
                position: relative;
                padding-bottom: 0;
                --gender-avatar-nudge-x: 52px;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) [data-testid="element-container"] {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) [data-testid="element-container"]:has(.gender-pick-cell) {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .gender-pick-cell {
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                gap: 0.12rem;
                margin: 0;
                transform: none;
                padding: 0.45rem 0.55rem;
                border-radius: 22px;
                border: 2px solid transparent;
                background: rgba(255, 255, 255, 0.2);
                box-sizing: border-box;
                transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease, transform 0.12s ease;
                cursor: pointer;
                user-select: none;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]):hover .gender-pick-cell {
                border-color: rgba(34, 197, 94, 0.5);
                box-shadow: 0 10px 30px rgba(20, 90, 70, 0.18);
                background: rgba(255, 255, 255, 0.65);
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]):has(button:active) .gender-pick-cell {
                transform: scale(0.985);
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]):has(button:focus-visible) .gender-pick-cell {
                border-color: rgba(22, 163, 74, 0.75);
                box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.35);
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .gender-inline-label {
                margin: 0;
                padding: 0;
                font-size: 1.62rem;
                font-weight: 800;
                color: #1f2937;
                text-align: center;
                line-height: 1.1;
                width: 100%;
                max-width: 100%;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .gender-pick-cell-emoji .gender-emoji-fallback {
                font-size: 6.55rem;
                line-height: 1;
                text-align: center;
                width: 288px;
                max-width: min(288px, 88vw);
                margin: 0 auto;
                padding: 0;
            }
            /* Transparent st.button on TOP of the card (later sibling + z-index) so the whole icon+label area hits the button */
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) [data-testid="element-container"]:has([data-testid="stButton"]) {
                position: absolute;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                width: 100% !important;
                max-width: none !important;
                height: 100% !important;
                min-height: 100% !important;
                /* Real hit target (avoid max() so older engines do not drop the whole rule) */
                min-height: 380px !important;
                min-width: min(288px, 88vw) !important;
                transform: none !important;
                z-index: 20;
                margin: 0 !important;
                padding: 0 !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                pointer-events: auto !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton {
                width: 100% !important;
                height: 100% !important;
                min-width: 0 !important;
                margin: 0 !important;
                pointer-events: auto !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton [data-baseweb="button"] {
                min-width: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton > button {
                width: 100% !important;
                height: 100% !important;
                min-width: 0 !important;
                min-height: 100% !important;
                max-height: none !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                border-width: 0 !important;
                box-shadow: none !important;
                background: rgba(0, 0, 0, 0) !important;
                background-color: rgba(0, 0, 0, 0) !important;
                color: transparent !important;
                font-size: 0 !important;
                line-height: 0 !important;
                opacity: 1 !important;
                cursor: pointer !important;
                outline: none !important;
                -webkit-appearance: none !important;
                appearance: none !important;
                overflow: hidden !important;
                text-indent: -9999px !important;
                white-space: nowrap !important;
                pointer-events: auto !important;
            }
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton > button:hover,
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton > button:focus,
            body:has(.assessment-gender-step) [data-testid="stVerticalBlock"]:not(:has([data-testid="stHorizontalBlock"])):has(.gender-pick-cell):has([data-testid="stButton"]) .stButton > button:focus-visible {
                background: rgba(0, 0, 0, 0) !important;
                background-color: rgba(0, 0, 0, 0) !important;
                opacity: 1 !important;
                box-shadow: none !important;
                outline: none !important;
            }
            .assessment-question {
                text-align: center;
                font-size: 2.05rem;
                font-weight: 750;
                color: #121826;
                margin-top: 0.15rem;
                margin-bottom: 0.35rem;
                letter-spacing: -0.01em;
            }
            .assessment-helper {
                text-align: center;
                color: #9aa3b2;
                font-size: 0.95rem;
                margin-bottom: 0.9rem;
            }
            .gender-label {
                text-align: center;
                font-size: 1.35rem;
                font-weight: 650;
                color: #1f2937;
                margin-bottom: 0.1rem;
            }
            .basic-info-wrap {
                max-width: 860px;
                margin: 0 auto;
                padding: 0 18px 8px 18px;
                transform: none;
            }
            body:has(.height-stage-lock) .assessment-shell {
                margin-top: 0 !important;
            }
            .height-top-back-wrap {
                display: flex;
                justify-content: flex-start;
                margin: 0 !important;
                position: fixed !important;
                top: 28px !important;
                left: 72px !important;
                z-index: 99999 !important;
                pointer-events: auto !important;
                transform: none !important;
            }
            .height-top-back-link {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 96px;
                min-height: 44px;
                padding: 0 18px;
                border-radius: 999px;
                background: #25d366;
                color: #ffffff !important;
                font-size: 1.08rem;
                font-weight: 900;
                text-decoration: none !important;
                letter-spacing: 0.01em;
                box-shadow: 0 6px 18px rgba(37, 211, 102, 0.28);
            }
            .height-top-back-link:hover {
                background: #22c55e;
                color: #ffffff !important;
            }
            .height-step-title {
                text-align: center;
                font-size: 3rem;
                font-weight: 900;
                color: #1f2a44;
                line-height: 1.06;
                margin: 0.1rem 0 0.2rem 0;
            }
            .height-step-subtitle {
                text-align: center;
                color: #a3acb8;
                font-size: 1.35rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }
            .height-ruler-stage {
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                margin-bottom: 0.6rem;
            }
            .height-ruler-column {
                width: 94px;
                height: 320px;
                border-radius: 4px;
                background:
                    repeating-linear-gradient(
                        180deg,
                        rgba(32, 181, 129, 0.28) 0px,
                        rgba(32, 181, 129, 0.28) 2px,
                        rgba(232, 250, 243, 1) 2px,
                        rgba(232, 250, 243, 1) 15px
                    );
                position: relative;
            }
            .height-ruler-mark {
                position: absolute;
                left: -56px;
                color: #6b7280;
                font-size: 1.95rem;
                font-weight: 800;
            }
            .height-ruler-mark.top { top: 18px; }
            .height-ruler-mark.mid { top: 138px; }
            .height-ruler-mark.bot { top: 258px; }
            .height-ruler-readout {
                color: #26314d;
                font-size: 3.6rem;
                line-height: 1;
                font-weight: 900;
                white-space: nowrap;
            }
            .height-ruler-readout small {
                font-size: 0.5em;
                font-weight: 800;
                margin-left: 6px;
            }
            .height-step-next .stButton > button {
                width: 100% !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-size: 1.95rem !important;
                font-weight: 900 !important;
                min-height: 68px !important;
                letter-spacing: 0.01em;
            }
            body:has(.height-stage-lock) [data-testid="stButton"] {
                display: flex !important;
                justify-content: center !important;
                margin-top: -28px !important;
                position: relative;
                z-index: 25;
            }
            body:has(.height-stage-lock) [data-testid="stButton"] > button {
                width: min(92%, 720px) !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-size: 2.45rem !important;
                font-weight: 900 !important;
                min-height: 56px !important;
                letter-spacing: 0.01em;
                box-shadow: none !important;
                text-shadow: none !important;
            }
            body:has(.height-stage-lock) [data-testid="stButton"] > button,
            body:has(.height-stage-lock) [data-testid="stButton"] > button * {
                color: #ffffff !important;
                font-weight: 900 !important;
            }
            body:has(.height-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.height-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.height-stage-lock) [data-testid="stButton"] > button:focus-visible {
                background: #22c55e !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            .height-next-link-wrap {
                width: min(92%, 720px);
                margin: 0 auto 0 auto;
                position: relative;
                z-index: 25;
            }

            /* Move the white card up without shifting back/next controls */
            body:has(.height-stage-lock) .assessment-card {
                margin-top: -60px !important;
            }

            /* Apply same upward card/button shift to other onboarding stages
               while keeping the Back link fixed/visible. */
            body:has(.weight-stage-lock) .assessment-card,
            body:has(.age-stage-lock) .assessment-card,
            body:has(.goal-stage-lock) .assessment-card,
            body:has(.goal-time-stage-lock) .assessment-card,
            body:has(.plan-intro-stage-lock) .assessment-card,
            body:has(.details-stage-lock) .assessment-card,
            body:has(.assessment-gender-step) .assessment-card {
                margin-top: -60px !important;
            }

            body:has(.weight-stage-lock) [data-testid="stButton"],
            body:has(.age-stage-lock) [data-testid="stButton"],
            body:has(.goal-stage-lock) [data-testid="stButton"],
            body:has(.goal-time-stage-lock) [data-testid="stButton"],
            body:has(.plan-intro-stage-lock) [data-testid="stButton"],
            body:has(.details-stage-lock) [data-testid="stButton"],
            body:has(.assessment-gender-step) [data-testid="stButton"] {
                display: flex !important;
                justify-content: center !important;
                margin-top: -28px !important;
                position: relative;
                z-index: 25;
            }
            .height-next-link-native {
                width: 100%;
                min-height: 56px;
                border-radius: 999px;
                background: #25d366;
                border: none;
                color: #ffffff !important;
                font-size: 2.45rem;
                font-weight: 900;
                letter-spacing: 0.01em;
                text-decoration: none !important;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            .height-next-link-native:hover,
            .height-next-link-native:focus,
            .height-next-link-native:focus-visible {
                background: #22c55e;
                color: #ffffff !important;
                outline: none;
            }
            .basic-info-title {
                text-align: center;
                font-size: 2rem;
                font-weight: 750;
                letter-spacing: -0.02em;
                color: #0f172a;
                margin-bottom: 0.12rem;
            }
            .basic-info-subtitle {
                text-align: center;
                font-size: 1rem;
                color: #5b6472;
                margin-bottom: 1rem;
            }
            .basic-info-card {
                background: #ffffff;
                border: none;
                border-radius: 28px;
                padding: 1.15rem 1.2rem 1.05rem 1.2rem;
                box-shadow: 0 18px 50px rgba(20, 90, 70, 0.1);
            }
            .weight-page-shell {
                max-width: 860px;
                margin: 0 auto;
                padding: 0 18px 10px 18px;
                transform: none;
            }
            .weight-card {
                background: transparent;
                border: none;
                border-radius: 0;
                box-shadow: none;
                padding: 0;
            }
            .weight-value {
                text-align: center;
                font-size: 3rem;
                line-height: 1;
                font-weight: 900;
                color: #26314d;
                margin: 0.65rem 0 0.45rem 0;
            }
            .weight-value small {
                font-size: 0.52em;
                font-weight: 750;
                color: #6b7280;
                margin-left: 5px;
            }
            .bmi-card {
                margin-top: 0.8rem;
                border: 1px solid rgba(31, 41, 55, 0.1);
                border-radius: 16px;
                padding: 0.72rem 0.8rem 0.85rem 0.8rem;
                background: #fbfcfc;
            }
            .bmi-head {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 8px;
                margin-bottom: 0.45rem;
            }
            .bmi-title {
                font-size: 1.15rem;
                font-weight: 800;
                color: #1f2937;
            }
            .bmi-score {
                font-size: 1.8rem;
                font-weight: 900;
                color: #1f2937;
            }
            .bmi-score span {
                font-size: 0.55em;
                font-weight: 700;
                margin-left: 4px;
                color: #16a34a;
            }
            .bmi-scale-4 {
                display: flex;
                width: 100%;
                height: 10px;
                border-radius: 999px;
                overflow: hidden;
                margin: 0.3rem 0 0.35rem 0;
            }
            .bmi-seg {
                flex: 1 1 0;
                height: 100%;
            }
            .bmi-seg-under { background: #7dd3fc; }
            .bmi-seg-ideal { background: #86efac; }
            .bmi-seg-over { background: #fde68a; }
            .bmi-seg-obese { background: #fca5a5; }
            .bmi-scale-labels {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 4px;
                margin-top: 0.1rem;
            }
            .bmi-scale-labels span {
                text-align: center;
                color: #9aa3b2;
                font-size: 0.92rem;
                font-weight: 700;
            }
            .weight-next-btn .stButton > button {
                width: min(92%, 720px) !important;
                margin: 0 auto !important;
                display: block !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-size: 1.95rem !important;
                font-weight: 900 !important;
                min-height: 62px !important;
                letter-spacing: 0 !important;
                margin-top: 1.05rem;
                box-shadow: none !important;
                text-shadow: none !important;
                padding: 0 !important;
            }
            .weight-next-btn .stButton > button:hover,
            .weight-next-btn .stButton > button:focus,
            .weight-next-btn .stButton > button:focus-visible {
                background: #22c55e !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            /* Cross-device fallback (no :has support): style onboarding Next buttons by wrapper classes */
            .weight-page-shell [data-testid="stButton"] {
                display: flex !important;
                justify-content: center !important;
            }
            .weight-page-shell [data-testid="stButton"] > button {
                width: min(92%, 720px) !important;
                margin: 0 auto !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-size: 1.95rem !important;
                font-weight: 900 !important;
                min-height: 62px !important;
                letter-spacing: 0 !important;
                box-shadow: none !important;
                text-shadow: none !important;
                margin-top: 1.05rem !important;
            }
            .weight-page-shell [data-testid="stButton"] > button,
            .weight-page-shell [data-testid="stButton"] > button * {
                color: #ffffff !important;
                font-weight: 900 !important;
            }
            .weight-page-shell [data-testid="stButton"] > button:hover,
            .weight-page-shell [data-testid="stButton"] > button:focus,
            .weight-page-shell [data-testid="stButton"] > button:focus-visible {
                background: #22c55e !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            .basic-info-wrap [data-testid="stFormSubmitButton"] {
                display: flex !important;
                justify-content: center !important;
            }
            .basic-info-wrap [data-testid="stFormSubmitButton"] button[kind="primary"] {
                width: min(92%, 720px) !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-weight: 900 !important;
                font-size: 1.95rem !important;
                min-height: 62px !important;
                margin-top: 1.05rem !important;
            }
            body:has(.weight-stage-lock) [data-testid="stButton"],
            body:has(.age-stage-lock) [data-testid="stButton"],
            body:has(.goal-stage-lock) [data-testid="stButton"],
            body:has(.goal-time-stage-lock) [data-testid="stButton"],
            body:has(.plan-intro-stage-lock) [data-testid="stButton"],
            body:has(.details-stage-lock) [data-testid="stButton"] {
                display: flex !important;
                justify-content: center !important;
            }
            body:has(.weight-stage-lock) [data-testid="stButton"] > button,
            body:has(.age-stage-lock) [data-testid="stButton"] > button,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button,
            body:has(.details-stage-lock) [data-testid="stButton"] > button {
                width: min(92%, 720px) !important;
                margin: 0 auto !important;
                border-radius: 999px !important;
                background: #25d366 !important;
                border: none !important;
                color: #ffffff !important;
                font-size: 1.95rem !important;
                font-weight: 900 !important;
                min-height: 62px !important;
                letter-spacing: 0 !important;
                box-shadow: none !important;
                text-shadow: none !important;
                margin-top: 1.05rem !important;
            }
            body:has(.weight-stage-lock) [data-testid="stButton"] > button,
            body:has(.weight-stage-lock) [data-testid="stButton"] > button *,
            body:has(.age-stage-lock) [data-testid="stButton"] > button,
            body:has(.age-stage-lock) [data-testid="stButton"] > button *,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button *,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button *,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button *,
            body:has(.details-stage-lock) [data-testid="stButton"] > button,
            body:has(.details-stage-lock) [data-testid="stButton"] > button * {
                color: #ffffff !important;
                font-weight: 900 !important;
                font-size: 2.15rem !important;
            }
            body:has(.weight-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.weight-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.weight-stage-lock) [data-testid="stButton"] > button:focus-visible,
            body:has(.age-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.age-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.age-stage-lock) [data-testid="stButton"] > button:focus-visible,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.goal-stage-lock) [data-testid="stButton"] > button:focus-visible,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.goal-time-stage-lock) [data-testid="stButton"] > button:focus-visible,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.plan-intro-stage-lock) [data-testid="stButton"] > button:focus-visible,
            body:has(.details-stage-lock) [data-testid="stButton"] > button:hover,
            body:has(.details-stage-lock) [data-testid="stButton"] > button:focus,
            body:has(.details-stage-lock) [data-testid="stButton"] > button:focus-visible {
                background: #22c55e !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
            body:has(.goal-time-stage-lock) .weight-next-btn {
                margin-top: 0.4rem !important;
            }
            body:has(.goal-time-stage-lock) .weight-next-btn .stButton > button {
                margin-top: 0.35rem !important;
            }
            .weight-bmi-footnote {
                text-align: center;
                color: #7a8a86;
                font-size: 0.96rem;
                font-weight: 600;
                line-height: 1.45;
                margin-top: 0.72rem;
            }
            body:has(.height-stage-lock),
            body:has(.height-stage-lock) html,
            body:has(.height-stage-lock) [data-testid="stAppViewContainer"],
            body:has(.height-stage-lock) [data-testid="stAppViewContainer"] > .main,
            body:has(.height-stage-lock) .main .block-container {
                height: auto !important;
                max-height: none !important;
                overflow: visible !important;
                overscroll-behavior: auto !important;
            }
            body:has(.height-stage-lock) .main .block-container {
                margin-top: 0 !important;
                padding-bottom: 1.25rem !important;
            }
            /* Keep height step centered in viewport (remove sidebar offset) */
            body:has(.height-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.height-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            /* Keep weight step centered in viewport (remove sidebar offset) */
            body:has(.weight-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.weight-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.weight-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            /* Keep age step centered in viewport (remove sidebar offset) */
            body:has(.age-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.age-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.age-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            /* Keep goal-weight step centered in viewport (remove sidebar offset) */
            body:has(.goal-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.goal-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.goal-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            /* Keep goal-time step centered in viewport (remove sidebar offset) */
            body:has(.goal-time-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.goal-time-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.goal-time-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            /* Keep plan-intro step centered in viewport (remove sidebar offset) */
            body:has(.plan-intro-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.plan-intro-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.plan-intro-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            body:has(.details-stage-lock) [data-testid="stSidebar"] {
                display: none !important;
                min-width: 0 !important;
                max-width: 0 !important;
                width: 0 !important;
            }
            body:has(.details-stage-lock) [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            body:has(.details-stage-lock) .main .block-container {
                padding-top: 0.35rem !important;
            }
            body:has(.details-stage-lock) .basic-info-wrap {
                max-width: 860px !important;
                margin: 0 auto !important;
                padding: 0 16px !important;
            }
            body:has(.details-stage-lock) .basic-info-card {
                background: rgba(255, 255, 255, 0.78) !important;
                border: 1px solid rgba(34, 197, 94, 0.26) !important;
                border-radius: 20px !important;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07) !important;
                padding: 1rem 1rem 1.2rem 1rem !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] {
                background: rgba(255, 255, 255, 0.86) !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                border-radius: 20px !important;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07) !important;
                padding: 16px 14px 14px 14px !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] label p,
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
                font-size: 1.2rem !important;
                color: #1f2937 !important;
                font-weight: 820 !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] {
                margin-bottom: 16px !important;
            }
            body:has(.details-stage-lock) [data-baseweb="select"] > div,
            body:has(.details-stage-lock) [data-baseweb="select"] input,
            body:has(.details-stage-lock) [data-baseweb="input"] > div,
            body:has(.details-stage-lock) [data-baseweb="input"] input {
                border-radius: 14px !important;
                border-color: rgba(34, 197, 94, 0.24) !important;
                background: #ffffff !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-baseweb="select"] > div,
            body:has(.details-stage-lock) [data-testid="stForm"] [data-baseweb="input"] > div,
            body:has(.details-stage-lock) [data-testid="stForm"] [data-baseweb="textarea"] > div {
                min-height: 54px !important;
                box-shadow: none !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-baseweb="tag"] {
                border-radius: 10px !important;
                background: #dcfce7 !important;
                color: #16a34a !important;
                border: 1px solid rgba(34, 197, 94, 0.35) !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [role="radiogroup"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 8px 10px !important;
                margin-top: 4px !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] {
                margin: 0 !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] > div {
                padding: 9px 14px !important;
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] label {
                font-size: 1rem !important;
                font-weight: 700 !important;
                color: #334155 !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] label > div:first-child {
                display: none !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] [aria-hidden="true"],
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] svg {
                display: none !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] label {
                gap: 0 !important;
                padding-left: 0 !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
            }
            body:has(.details-stage-lock) [data-testid="stRadio"] [data-baseweb="radio"] input[type="radio"] {
                position: absolute !important;
                opacity: 0 !important;
                width: 0 !important;
                height: 0 !important;
                pointer-events: none !important;
            }
            /* Render health conditions with checkbox-pills that match radio pill style */
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] {
                margin: 0 0 8px 0 !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] label {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                min-height: 40px !important;
                width: 100% !important;
                padding: 9px 14px !important;
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
                cursor: pointer !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] label > div:first-child,
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child,
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] [role="checkbox"],
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] svg {
                display: none !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] label p {
                margin: 0 !important;
                font-size: 1rem !important;
                font-weight: 800 !important;
                color: #334155 !important;
                line-height: 1.25 !important;
                text-align: center !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"]:has(input:checked) label {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"]:has(input:checked) label p {
                color: #166534 !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] [data-testid="stCheckbox"] input[type="checkbox"] {
                position: absolute !important;
                opacity: 0 !important;
                width: 0 !important;
                height: 0 !important;
                pointer-events: none !important;
                display: none !important;
            }
            /* Match multi-select pills to the same style as other question choices */
            body:has(.details-stage-lock) [data-testid="stPills"] [data-baseweb="button-group"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 8px 10px !important;
                margin-top: 4px !important;
            }
            body:has(.details-stage-lock) [data-testid="stPills"] button {
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                padding: 9px 14px !important;
                min-height: 40px !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            }
            body:has(.details-stage-lock) [data-testid="stPills"] button[aria-pressed="true"],
            body:has(.details-stage-lock) [data-testid="stPills"] button[data-selected="true"] {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
                color: #166534 !important;
            }
            /* Fallback selectors for browsers/DOM variants */
            .basic-info-wrap [data-testid*="stPill"] [data-baseweb="button-group"],
            .basic-info-wrap [data-testid="stPills"] [data-baseweb="button-group"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 8px 10px !important;
                margin-top: 4px !important;
            }
            .basic-info-wrap [data-testid*="stPill"] button,
            .basic-info-wrap [data-testid="stPills"] button {
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                padding: 9px 14px !important;
                min-height: 40px !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            }
            .basic-info-wrap [data-testid*="stPill"] [role="checkbox"],
            .basic-info-wrap [data-testid*="stPill"] [role="radio"],
            .basic-info-wrap [data-testid*="stPill"] [role="option"],
            .basic-info-wrap [data-testid*="stPill"] [data-baseweb="button"] {
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            }
            .basic-info-wrap [data-testid*="stPill"] button[aria-pressed="true"],
            .basic-info-wrap [data-testid*="stPill"] button[data-selected="true"],
            .basic-info-wrap [data-testid="stPills"] button[aria-pressed="true"],
            .basic-info-wrap [data-testid="stPills"] button[data-selected="true"] {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
                color: #166534 !important;
            }
            .basic-info-wrap [data-testid*="stPill"] [aria-selected="true"],
            .basic-info-wrap [data-testid*="stPill"] [aria-checked="true"],
            .basic-info-wrap [data-testid*="stPill"] [data-baseweb="button"][data-selected="true"],
            .basic-info-wrap [data-testid*="stPill"] [data-baseweb="button"][aria-pressed="true"] {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
                color: #166534 !important;
            }
            .basic-info-wrap [data-testid*="stPill"] * {
                color: inherit !important;
            }
            /* Hard fallback for DOM variants: style all option-like buttons in details form */
            body:has(.details-stage-lock) [data-testid="stForm"] button:not([kind]) {
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.28) !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                padding: 9px 14px !important;
                min-height: 40px !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
            }
            body:has(.details-stage-lock) [data-testid="stForm"] button:not([kind])[aria-pressed="true"],
            body:has(.details-stage-lock) [data-testid="stForm"] button:not([kind])[aria-selected="true"],
            body:has(.details-stage-lock) [data-testid="stForm"] button:not([kind])[aria-checked="true"],
            body:has(.details-stage-lock) [data-testid="stForm"] [aria-pressed="true"] > button:not([kind]),
            body:has(.details-stage-lock) [data-testid="stForm"] [aria-selected="true"] > button:not([kind]),
            body:has(.details-stage-lock) [data-testid="stForm"] [aria-checked="true"] > button:not([kind]) {
                background: #dcfce7 !important;
                border-color: rgba(34, 197, 94, 0.68) !important;
                color: #166534 !important;
            }
            .basic-info-wrap [data-testid="stMultiSelect"] [data-baseweb="tag"] {
                border-radius: 999px !important;
                border: 1px solid rgba(34, 197, 94, 0.55) !important;
                background: #dcfce7 !important;
                color: #166534 !important;
                font-weight: 700 !important;
            }
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button[kind="primary"] {
                background: #25d366 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 999px !important;
                min-height: 62px !important;
                font-size: 2.15rem !important;
                font-weight: 900 !important;
                letter-spacing: 0 !important;
                text-shadow: none !important;
            }
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button[kind="secondary"] {
                background: #ffffff !important;
                color: #16a34a !important;
                border: 2px solid rgba(34, 197, 94, 0.55) !important;
                border-radius: 999px !important;
                min-height: 56px !important;
                font-size: 1.35rem !important;
                font-weight: 850 !important;
            }
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {
                filter: brightness(0.98) !important;
            }
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button {
                background: #25d366 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 999px !important;
                min-height: 54px !important;
                font-size: 2.15rem !important;
                font-weight: 850 !important;
            }
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button,
            body:has(.details-stage-lock) [data-testid="stFormSubmitButton"] button * {
                color: #ffffff !important;
                font-size: 2.15rem !important;
                font-weight: 900 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if show_gender_page:
        male_path, female_path = resolve_gender_avatar_paths()
        st.markdown('<div class="assessment-shell">', unsafe_allow_html=True)
        st.markdown(
            '<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_welcome_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            textwrap.dedent(
                """
                <div class="assessment-card assessment-gender-step">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:12%;"></div></div>
                    <div class="assessment-question">What is your gender?</div>
                    <div class="assessment-helper">Biological sex can influence metabolism and diet strategy.</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gender-pick-gap"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            if female_path is not None:
                _f_html = _gender_pick_cell_html_from_pil(
                    _gender_avatar_pil_transparent(
                        str(female_path.resolve()),
                        female_path.stat().st_mtime,
                    ),
                    "Female",
                    _ONBOARDING_GENDER_AVATAR_WIDTH_PX,
                    "F",
                )
            else:
                _f_html = _gender_pick_cell_html_emoji("👩", "Female", _ONBOARDING_GENDER_AVATAR_WIDTH_PX, "F")
            st.markdown(_f_html, unsafe_allow_html=True)
        with c2:
            if male_path is not None:
                _m_html = _gender_pick_cell_html_from_pil(
                    _gender_avatar_pil_transparent(
                        str(male_path.resolve()),
                        male_path.stat().st_mtime,
                    ),
                    "Male",
                    _ONBOARDING_GENDER_AVATAR_WIDTH_PX,
                    "M",
                )
            else:
                _m_html = _gender_pick_cell_html_emoji("👨", "Male", _ONBOARDING_GENDER_AVATAR_WIDTH_PX, "M")
            st.markdown(_m_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        return None

    if "onboarding_stage" not in st.session_state:
        st.session_state["onboarding_stage"] = "height"
    if "onboarding_height_cm" not in st.session_state:
        st.session_state["onboarding_height_cm"] = int(defaults["height_cm"])
    if "onboarding_weight_kg" not in st.session_state:
        st.session_state["onboarding_weight_kg"] = float(defaults["weight_kg"])
    if "onboarding_goal_weight_kg" not in st.session_state:
        st.session_state["onboarding_goal_weight_kg"] = float(defaults["goal_weight_kg"])
    if "onboarding_goal_year" not in st.session_state:
        target_seed = date.today() + timedelta(weeks=12)
        st.session_state["onboarding_goal_year"] = int(target_seed.year)
        st.session_state["onboarding_goal_month"] = int(target_seed.month)
        st.session_state["onboarding_goal_day"] = int(target_seed.day)
    if "onboarding_age" not in st.session_state:
        st.session_state["onboarding_age"] = int(defaults["age"])

    if st.session_state["onboarding_stage"] == "height":
        current_height = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))

        st.markdown('<div class="height-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="assessment-shell">', unsafe_allow_html=True)
        st.markdown(
            '<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                <div class="assessment-card assessment-gender-step">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:24%;"></div></div>
                    <div class="assessment-question">What is your height?</div>
                    <div class="assessment-helper">Accurate height helps us calculate your BMI.</div>
                </div>
                """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        selected_sex_for_link = st.session_state.get("onboarding_selected_sex", "M")
        ruler_component_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                }}
                .stage {{
                    min-height: 220px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 20px;
                }}
                .col {{
                    position: relative;
                    width: 94px;
                    height: 360px;
                    border-radius: 4px;
                    background: rgba(235, 251, 244, 0.95);
                    cursor: ns-resize;
                    user-select: none;
                    touch-action: none;
                    overflow: hidden;
                }}
                .track {{
                    position: absolute;
                    left: 0;
                    right: 0;
                    top: 0;
                    transform: translateY(0);
                    will-change: transform;
                }}
                .tick {{
                    position: absolute;
                    left: 8px;
                    right: 6px;
                    border-top: 1px solid rgba(37, 164, 124, 0.34);
                }}
                .tick.major {{
                    border-top: 2px solid rgba(37, 164, 124, 0.56);
                }}
                .tick .tick-label {{
                    position: absolute;
                    left: 2px;
                    top: -13px;
                    color: rgba(66, 82, 110, 0.92);
                    font-size: 1.02rem;
                    font-weight: 850;
                    line-height: 1;
                    background: rgba(255, 255, 255, 0.52);
                    border-radius: 5px;
                    padding: 1px 3px;
                }}
                .pointer {{
                    position: absolute;
                    left: -2px;
                    width: 76px;
                    height: 10px;
                    border-radius: 999px;
                    background: #27d978;
                    top: 170px;
                    pointer-events: none;
                }}
                .readout {{
                    color: #26314d;
                    font-size: 3.6rem;
                    line-height: 1;
                    font-weight: 900;
                    white-space: nowrap;
                    min-width: 210px;
                    margin-left: 18px;
                }}
                .readout small {{
                    font-size: 0.5em;
                    font-weight: 800;
                    margin-left: 6px;
                }}
                .next-wrap {{
                    margin-top: 44px;
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }}
                .next-btn {{
                    width: min(92%, 720px);
                    border: none;
                    border-radius: 999px;
                    background: #25d366;
                    color: #ffffff;
                    font-size: 1.95rem;
                    font-weight: 900;
                    min-height: 56px;
                    letter-spacing: 0.01em;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                }}
            </style>
        </head>
        <body>
            <div>
                <div class="stage">
                    <div id="ruler" class="col">
                        <div id="track" class="track"></div>
                        <div id="pointer" class="pointer"></div>
                    </div>
                    <div id="readout" class="readout">{current_height}<small>cm</small></div>
                </div>
            </div>
            <script>
                const minH = 145, maxH = 230;
                const ruler = document.getElementById("ruler");
                const track = document.getElementById("track");
                const readout = document.getElementById("readout");
                let current = {current_height};
                let dragging = false;
                let dragStartY = 0;
                let dragStartHeight = current;
                const pxPerCm = 10;
                const pointerTop = 170;
                const pointerHeight = 10;
                const centerY = pointerTop + (pointerHeight / 2);
                const trackPad = 260;
                const trackHeight = trackPad * 2 + (maxH - minH) * pxPerCm;

                function clamp(v, lo, hi) {{
                    return Math.max(lo, Math.min(hi, v));
                }}

                function yForHeight(h) {{
                    return trackPad + (maxH - h) * pxPerCm;
                }}

                function buildTrack() {{
                    track.style.height = `${{trackHeight}}px`;
                    const parts = [];
                    for (let h = minH; h <= maxH; h += 1) {{
                        const y = yForHeight(h);
                        const major = h % 10 === 0;
                        const label = major ? `<span class="tick-label">${{h}}</span>` : "";
                        parts.push(
                            `<div class="tick ${{major ? "major" : "minor"}}" style="top:${{Math.round(y)}}px">${{label}}</div>`
                        );
                    }}
                    track.innerHTML = parts.join("");
                }}

                function render(h) {{
                    current = clamp(Math.round(h), minH, maxH);
                    const translateY = centerY - yForHeight(current);
                    track.style.transform = `translateY(${{Math.round(translateY)}}px)`;
                    readout.innerHTML = `${{current}}<small>cm</small>`;
                    try {{
                        window.parent.postMessage({{
                            type: "onboarding-height",
                            value: current,
                            sex: "{selected_sex_for_link}",
                        }}, "*");
                    }} catch (e) {{
                        // Ignore bridge failures.
                    }}
                }}

                function onMoveClientY(clientY) {{
                    if (!dragging) return;
                    const deltaCm = (clientY - dragStartY) / pxPerCm;
                    const h = dragStartHeight + deltaCm;
                    render(h);
                }}

                function startDrag(clientY) {{
                    dragging = true;
                    dragStartY = clientY;
                    dragStartHeight = current;
                }}

                function endDrag(clientY = null) {{
                    if (!dragging) return;
                    if (clientY !== null) {{
                        const deltaCm = (clientY - dragStartY) / pxPerCm;
                        render(dragStartHeight + deltaCm);
                    }}
                    dragging = false;
                }}

                // Pointer events (modern browsers)
                ruler.addEventListener("pointerdown", (e) => {{
                    startDrag(e.clientY);
                }});
                window.addEventListener("pointermove", (e) => onMoveClientY(e.clientY));
                window.addEventListener("pointerup", (e) => endDrag(e.clientY));
                window.addEventListener("pointercancel", () => endDrag());

                // Mouse fallback
                ruler.addEventListener("mousedown", (e) => {{
                    e.preventDefault();
                    startDrag(e.clientY);
                }});
                window.addEventListener("mousemove", (e) => onMoveClientY(e.clientY));
                window.addEventListener("mouseup", (e) => endDrag(e.clientY));

                // Touch fallback
                ruler.addEventListener("touchstart", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    startDrag(e.touches[0].clientY);
                }}, {{ passive: true }});
                window.addEventListener("touchmove", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    onMoveClientY(e.touches[0].clientY);
                }}, {{ passive: true }});
                window.addEventListener("touchend", (e) => {{
                    if (!e.changedTouches || !e.changedTouches.length) {{
                        endDrag();
                        return;
                    }}
                    endDrag(e.changedTouches[0].clientY);
                }}, {{ passive: true }});
                // Only prevent wheel default while actively dragging the ruler
                window.addEventListener("wheel", (e) => {{
                    if (dragging) e.preventDefault();
                }}, {{ passive: false }});

                buildTrack();
                render(current);
            </script>
        </body>
        </html>
        """
        components.html(ruler_component_html, height=320, scrolling=False)
        st.markdown(
            f'<div class="height-next-link-wrap"><a id="height-next-link" class="height-next-link-native" href="?onboarding_sex={selected_sex_for_link}&onboarding_height={current_height}&onboarding_next=1" target="_self" rel="noopener">Next</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)  # .assessment-shell
        return None

    if st.session_state["onboarding_stage"] == "weight":
        current_weight = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
        height_for_bmi = float(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
        bmi = current_weight / ((height_for_bmi / 100) ** 2)
        bmi_label = bmi_status_label(bmi)

        st.markdown('<div class="weight-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="weight-page-shell">', unsafe_allow_html=True)
        selected_sex_for_back = st.session_state.get("onboarding_selected_sex", "M")
        st.markdown(
            f'<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_weight_back=1&onboarding_sex={selected_sex_for_back}" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weight-card">', unsafe_allow_html=True)
        st.markdown(
            """
                <div class="assessment-card" style="margin-top:0; margin-bottom:10px; box-shadow:none; padding:1.15rem 0.2rem 0.8rem 0.2rem;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:36%;"></div></div>
                    <div class="assessment-question">What is your weight?</div>
                    <div class="assessment-helper">Accurate body data helps us calculate your BMI.</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        weight_ruler_component_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                }}
                .stage {{
                    min-height: 150px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                }}
                .live-value {{
                    text-align: center;
                    font-size: 2.2rem;
                    line-height: 1;
                    font-weight: 900;
                    color: #26314d;
                    margin: 0.1rem 0 0.5rem 0;
                }}
                .live-value small {{
                    font-size: 0.52em;
                    font-weight: 750;
                    color: #6b7280;
                    margin-left: 5px;
                }}
                .live-bmi-card {{
                    margin-top: 0.65rem;
                    border: 1px solid rgba(31, 41, 55, 0.1);
                    border-radius: 14px;
                    padding: 0.6rem 0.75rem 0.65rem 0.75rem;
                    background: #fbfcfc;
                    width: min(92vw, 760px);
                }}
                .live-bmi-head {{
                    display: flex;
                    justify-content: space-between;
                    align-items: baseline;
                    gap: 8px;
                    margin-bottom: 0.42rem;
                }}
                .live-bmi-title {{
                    font-size: 1.02rem;
                    font-weight: 800;
                    color: #1f2937;
                }}
                .live-bmi-score {{
                    font-size: 1.45rem;
                    font-weight: 900;
                    color: #1f2937;
                }}
                .live-bmi-score span {{
                    font-size: 0.62em;
                    font-weight: 800;
                    margin-left: 4px;
                    color: #16a34a;
                }}
                .live-bmi-scale {{
                    display: flex;
                    width: 100%;
                    height: 10px;
                    border-radius: 999px;
                    overflow: hidden;
                    margin: 0.15rem 0 0.35rem 0;
                }}
                .live-bmi-seg {{
                    flex: 1 1 0;
                    height: 100%;
                }}
                .live-bmi-seg-under {{ background: #7dd3fc; }}
                .live-bmi-seg-ideal {{ background: #86efac; }}
                .live-bmi-seg-over {{ background: #fde68a; }}
                .live-bmi-seg-obese {{ background: #fca5a5; }}
                .live-bmi-labels {{
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                    gap: 4px;
                    margin-top: 0.08rem;
                }}
                .live-bmi-labels span {{
                    text-align: center;
                    color: #9aa3b2;
                    font-size: 0.86rem;
                    font-weight: 700;
                }}
                .ruler {{
                    position: relative;
                    width: min(92vw, 760px);
                    height: 70px;
                    border-radius: 12px;
                    background: rgba(235, 251, 244, 0.95);
                    overflow: hidden;
                    cursor: ew-resize;
                    user-select: none;
                    touch-action: none;
                }}
                .track {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 100%;
                    transform: translateX(0);
                    will-change: transform;
                }}
                .tick {{
                    position: absolute;
                    bottom: 22px;
                    width: 1px;
                    background: rgba(37, 164, 124, 0.36);
                    height: 18px;
                }}
                .tick.major {{
                    height: 30px;
                    background: rgba(37, 164, 124, 0.58);
                }}
                .tick .tick-label {{
                    position: absolute;
                    top: 34px;
                    left: -12px;
                    min-width: 24px;
                    text-align: center;
                    color: rgba(66, 82, 110, 0.92);
                    font-size: 0.98rem;
                    font-weight: 850;
                    line-height: 1;
                }}
                .pointer {{
                    position: absolute;
                    top: 12px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 10px;
                    height: 34px;
                    border-radius: 999px;
                    background: #27d978;
                    pointer-events: none;
                }}
                .next-wrap {{
                    margin-top: 14px;
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }}
                .next-btn {{
                    width: min(92%, 720px);
                    border: none;
                    border-radius: 999px;
                    background: #25d366;
                    color: #ffffff;
                    font-size: 1.95rem;
                    font-weight: 900;
                    min-height: 56px;
                    letter-spacing: 0.01em;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                }}
            </style>
        </head>
        <body>
            <div class="stage">
                <div id="liveWeight" class="live-value"></div>
                <div id="ruler" class="ruler">
                    <div id="track" class="track"></div>
                    <div class="pointer"></div>
                </div>
                <div class="live-bmi-card">
                    <div class="live-bmi-head">
                        <div class="live-bmi-title">Your BMI</div>
                        <div id="liveBmiScore" class="live-bmi-score"></div>
                    </div>
                    <div class="live-bmi-scale">
                        <div class="live-bmi-seg live-bmi-seg-under"></div>
                        <div class="live-bmi-seg live-bmi-seg-ideal"></div>
                        <div class="live-bmi-seg live-bmi-seg-over"></div>
                        <div class="live-bmi-seg live-bmi-seg-obese"></div>
                    </div>
                    <div class="live-bmi-labels">
                        <span>Underweight</span>
                        <span>Ideal</span>
                        <span>Overweight</span>
                        <span>Obese</span>
                    </div>
                </div>
            </div>
            <script>
                const minW = 35.0, maxW = 250.0;
                const step = 0.1;
                const majorEvery = 5.0;
                const heightCm = {height_for_bmi:.1f};
                const ruler = document.getElementById("ruler");
                const track = document.getElementById("track");
                const liveWeight = document.getElementById("liveWeight");
                const liveBmiScore = document.getElementById("liveBmiScore");
                let current = {current_weight:.1f};
                let dragging = false;
                let dragStartX = 0;
                let dragStartWeight = current;
                let lastCommitted = current;

                const pxPerKg = 22;
                const trackPad = 460;
                const rulerWidth = () => ruler.clientWidth || 760;
                const centerX = () => rulerWidth() / 2;
                const trackWidth = () => trackPad * 2 + (maxW - minW) * pxPerKg;

                function clamp(v, lo, hi) {{
                    return Math.max(lo, Math.min(hi, v));
                }}
                function round1(v) {{
                    return Math.round(v * 10) / 10;
                }}
                function xForWeight(w) {{
                    return trackPad + (w - minW) * pxPerKg;
                }}
                function buildTrack() {{
                    track.style.width = `${{Math.round(trackWidth())}}px`;
                    const parts = [];
                    for (let w = minW; w <= maxW + 0.0001; w += 0.5) {{
                        const ww = Math.round(w * 10) / 10;
                        const x = xForWeight(ww);
                        const isMajor = Math.abs((ww / majorEvery) - Math.round(ww / majorEvery)) < 0.0001;
                        const label = isMajor ? `<span class="tick-label">${{Math.round(ww)}}</span>` : "";
                        parts.push(
                            `<div class="tick ${{isMajor ? "major" : "minor"}}" style="left:${{Math.round(x)}}px">${{label}}</div>`
                        );
                    }}
                    track.innerHTML = parts.join("");
                }}
                function render(w) {{
                    current = clamp(round1(w), minW, maxW);
                    const tx = centerX() - xForWeight(current);
                    track.style.transform = `translateX(${{Math.round(tx)}}px)`;

                    const bmi = current / Math.pow(heightCm / 100, 2);
                    let bmiLabel = "Obese";
                    let bmiColor = "#fca5a5";
                    if (bmi < 18.5) bmiLabel = "Underweight";
                    else if (bmi < 25) bmiLabel = "Ideal";
                    else if (bmi < 30) bmiLabel = "Overweight";
                    if (bmiLabel === "Underweight") bmiColor = "#7dd3fc";
                    else if (bmiLabel === "Ideal") bmiColor = "#86efac";
                    else if (bmiLabel === "Overweight") bmiColor = "#fde68a";
                    if (liveWeight) liveWeight.innerHTML = `${{current.toFixed(1)}}<small>kg</small>`;
                    if (liveBmiScore) liveBmiScore.innerHTML = `${{bmi.toFixed(1)}}<span style="color:${{bmiColor}}">${{bmiLabel}}</span>`;
                    try {{
                        window.parent.postMessage({{
                            type: "onboarding-weight",
                            value: current,
                            sex: "{selected_sex_for_back}",
                        }}, "*");
                    }} catch (e) {{
                        // Ignore bridge failures.
                    }}
                }}
                function commitWeightToApp() {{
                    // Avoid top-level URL mutations from inside iframe.
                    // They can trigger Streamlit removeChild errors on some browsers.
                    return;
                }}
                function onMoveClientX(clientX) {{
                    if (!dragging) return;
                    const deltaKg = (clientX - dragStartX) / pxPerKg;
                    render(dragStartWeight - deltaKg);
                }}
                function startDrag(clientX) {{
                    dragging = true;
                    dragStartX = clientX;
                    dragStartWeight = current;
                }}
                function endDrag(clientX = null) {{
                    if (!dragging) return;
                    if (clientX !== null) {{
                        const deltaKg = (clientX - dragStartX) / pxPerKg;
                        render(dragStartWeight - deltaKg);
                    }}
                    dragging = false;
                    if (Math.abs(current - lastCommitted) >= 0.1) {{
                        lastCommitted = current;
                        commitWeightToApp();
                    }}
                }}

                ruler.addEventListener("pointerdown", (e) => startDrag(e.clientX));
                window.addEventListener("pointermove", (e) => onMoveClientX(e.clientX));
                window.addEventListener("pointerup", (e) => endDrag(e.clientX));
                window.addEventListener("pointercancel", () => endDrag());

                ruler.addEventListener("mousedown", (e) => {{
                    e.preventDefault();
                    startDrag(e.clientX);
                }});
                window.addEventListener("mousemove", (e) => onMoveClientX(e.clientX));
                window.addEventListener("mouseup", (e) => endDrag(e.clientX));

                ruler.addEventListener("touchstart", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    startDrag(e.touches[0].clientX);
                }}, {{ passive: true }});
                window.addEventListener("touchmove", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    onMoveClientX(e.touches[0].clientX);
                }}, {{ passive: true }});
                window.addEventListener("touchend", (e) => {{
                    if (!e.changedTouches || !e.changedTouches.length) {{
                        endDrag();
                        return;
                    }}
                    endDrag(e.changedTouches[0].clientX);
                }}, {{ passive: true }});

                window.addEventListener("wheel", (e) => {{
                    if (dragging) e.preventDefault();
                }}, {{ passive: false }});

                window.addEventListener("resize", () => render(current));
                buildTrack();
                render(current);
            </script>
        </body>
        </html>
        """
        components.html(weight_ruler_component_html, height=220, scrolling=False)
        st.markdown(
            f'<div class="weight-next-btn"><a id="weight-next-link" class="height-next-link-native" href="?onboarding_sex={selected_sex_for_back}&onboarding_height={int(round(height_for_bmi))}&onboarding_weight={current_weight:.1f}&onboarding_weight_next=1" target="_self" rel="noopener">Next</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                <div class="weight-bmi-footnote">
                    Note: BMI is a commonly used screening indicator for obesity diagnosis.<br/>
                    BMI (kg/m²) = Weight (kg) / Height² (m²)
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-card
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-page-shell
        return None

    if st.session_state["onboarding_stage"] == "age":
        today = date.today()
        default_year = int(st.session_state.get("onboarding_birth_year", today.year - int(defaults["age"])))
        default_month = int(st.session_state.get("onboarding_birth_month", 1))
        default_day = int(st.session_state.get("onboarding_birth_day", 1))
        current_height_for_back = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
        current_weight_for_back = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
        selected_sex_for_age_back = st.session_state.get("onboarding_selected_sex", "M")

        st.markdown('<div class="age-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="weight-page-shell">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_sex={selected_sex_for_age_back}&onboarding_height={current_height_for_back}&onboarding_weight={current_weight_for_back:.1f}&onboarding_age_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weight-card">', unsafe_allow_html=True)
        st.markdown(
            """
                <div class="assessment-card" style="margin-top:0; margin-bottom:10px; box-shadow:none; padding:1.15rem 0.2rem 0.8rem 0.2rem;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:48%;"></div></div>
                    <div class="assessment-question">What is your birth date?</div>
                    <div class="assessment-helper">Age and metabolism are closely related.</div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        age_wheel_component_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                }}
                .wrap {{
                    width: min(94vw, 760px);
                    margin: 0 auto;
                }}
                .wheels {{
                    display: grid;
                    grid-template-columns: 1.65fr 0.9fr 1fr;
                    gap: 10px;
                    align-items: center;
                    margin-top: 4px;
                }}
                .wheel {{
                    height: 210px;
                    overflow-y: auto;
                    scroll-snap-type: y mandatory;
                    scroll-behavior: smooth;
                    -webkit-overflow-scrolling: touch;
                    scrollbar-width: none;
                    -ms-overflow-style: none;
                    background: #f8fafb;
                    border-radius: 14px;
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    position: relative;
                    cursor: grab;
                }}
                .wheel.dragging {{
                    cursor: grabbing;
                }}
                .wheel::-webkit-scrollbar {{ display: none; }}
                .wheel::before {{
                    content: none;
                }}
                .item {{
                    height: 54px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.18rem;
                    font-weight: 700;
                    color: rgba(15, 23, 42, 0.34);
                    scroll-snap-align: center;
                    user-select: none;
                    position: relative;
                    z-index: 3;
                    transition: color 120ms ease, font-size 120ms ease, font-weight 120ms ease;
                }}
                .item.active {{
                    color: #1f2937;
                    font-size: 2rem;
                    font-weight: 900;
                    background: rgba(15, 23, 42, 0.06);
                    border-radius: 10px;
                }}
                .age-card {{
                    margin-top: 30px;
                    border: 1px solid rgba(31, 41, 55, 0.1);
                    border-radius: 16px;
                    background: #fbfcfc;
                    padding: 10px 14px;
                }}
                .age-top {{
                    display: flex;
                    justify-content: space-between;
                    align-items: baseline;
                }}
                .age-top .title {{
                    font-size: 1.05rem;
                    font-weight: 800;
                    color: #1f2937;
                }}
                .age-top .score {{
                    font-size: 2rem;
                    font-weight: 900;
                    color: #1f2937;
                }}
                .age-top .score span {{
                    font-size: 0.5em;
                    color: #16a34a;
                    margin-left: 4px;
                }}
                .age-note {{
                    margin-top: 6px;
                    color: #4b5563;
                    font-size: 0.96rem;
                    line-height: 1.42;
                    font-weight: 650;
                }}
                .next {{
                    margin-top: 30px;
                    width: 100%;
                    min-height: 62px;
                    border-radius: 999px;
                    background: #25d366;
                    color: #fff !important;
                    font-size: 2rem;
                    font-weight: 900;
                    text-decoration: none !important;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                }}
            </style>
        </head>
        <body>
            <div class="wrap">
                <div class="wheels">
                    <div id="wheel-month" class="wheel"></div>
                    <div id="wheel-day" class="wheel"></div>
                    <div id="wheel-year" class="wheel"></div>
                </div>
                <div class="age-card">
                    <div class="age-top">
                        <div class="title">Your Age</div>
                        <div id="age-score" class="score">0<span>years</span></div>
                    </div>
                    <div class="age-note">Higher baseline metabolism and activity response supports structured weight management.</div>
                </div>
            </div>
            <script>
                const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
                const days = Array.from({{length: 31}}, (_, i) => i + 1);
                const currentYear = new Date().getFullYear();
                const years = [];
                for (let y = currentYear - 15; y >= currentYear - 75; y--) years.push(y);

                let month = {default_month};
                let day = {default_day};
                let year = {default_year};

                const wm = document.getElementById("wheel-month");
                const wd = document.getElementById("wheel-day");
                const wy = document.getElementById("wheel-year");
                const ageScore = document.getElementById("age-score");

                function makeItems(wheel, values, labeler) {{
                    wheel.innerHTML = "";
                    for (let i = 0; i < 2; i++) {{
                        const pad = document.createElement("div");
                        pad.className = "item";
                        pad.textContent = "";
                        wheel.appendChild(pad);
                    }}
                    values.forEach(v => {{
                        const d = document.createElement("div");
                        d.className = "item";
                        d.dataset.value = String(v);
                        d.textContent = labeler(v);
                        wheel.appendChild(d);
                    }});
                    for (let i = 0; i < 2; i++) {{
                        const pad = document.createElement("div");
                        pad.className = "item";
                        pad.textContent = "";
                        wheel.appendChild(pad);
                    }}
                }}
                function scrollToValue(wheel, values, value) {{
                    const idx = values.indexOf(value);
                    const row = 54;
                    const target = Math.max(0, (idx + 2) * row - (wheel.clientHeight / 2) + (row / 2));
                    wheel.scrollTop = target;
                }}
                function nearestValue(wheel, values) {{
                    const row = 54;
                    const center = wheel.scrollTop + wheel.clientHeight / 2;
                    const idx = Math.round((center - row / 2) / row) - 2;
                    const clamped = Math.max(0, Math.min(values.length - 1, idx));
                    return values[clamped];
                }}
                function markActive(wheel, values, selected) {{
                    wheel.querySelectorAll(".item").forEach(el => {{
                        const v = el.dataset.value;
                        el.classList.toggle("active", v !== undefined && String(selected) === v);
                    }});
                }}
                function daysInMonth(y, m) {{
                    return new Date(y, m, 0).getDate();
                }}
                function safeDate(y, m, d) {{
                    const md = daysInMonth(y, m);
                    const sd = Math.min(d, md);
                    return new Date(y, m - 1, sd);
                }}
                function calcAge(y, m, d) {{
                    const today = new Date();
                    const dob = safeDate(y, m, d);
                    let age = today.getFullYear() - dob.getFullYear();
                    const beforeBirthday =
                        (today.getMonth() + 1 < m) ||
                        ((today.getMonth() + 1 === m) && (today.getDate() < d));
                    if (beforeBirthday) age -= 1;
                    age = Math.max(16, Math.min(75, age));
                    return age;
                }}
                function refresh() {{
                    const maxDay = daysInMonth(year, month);
                    if (day > maxDay) day = maxDay;
                    const age = calcAge(year, month, day);
                    markActive(wm, months.map((_, i) => i + 1), month);
                    markActive(wd, days, day);
                    markActive(wy, years, year);
                    ageScore.innerHTML = `${{age}}<span>years</span>`;
                }}
                function bind(wheel, values, onPick) {{
                    let t = null;
                    const row = 54;
                    const snapToNearest = () => {{
                        const center = wheel.scrollTop + wheel.clientHeight / 2;
                        const idx = Math.round((center - row / 2) / row) - 2;
                        const clamped = Math.max(0, Math.min(values.length - 1, idx));
                        const target = Math.max(0, (clamped + 2) * row - (wheel.clientHeight / 2) + (row / 2));
                        wheel.scrollTo({{ top: target, behavior: "smooth" }});
                        onPick(values[clamped]);
                        refresh();
                    }};
                    const onScroll = () => {{
                        if (t) window.clearTimeout(t);
                        t = window.setTimeout(() => {{
                            snapToNearest();
                        }}, 90);
                    }};
                    wheel.addEventListener("scroll", onScroll, {{ passive: true }});

                    // Drag-to-scroll support for desktop and touch-pointer devices
                    let isDragging = false;
                    let startY = 0;
                    let startScrollTop = 0;
                    const onPointerDown = (e) => {{
                        isDragging = true;
                        startY = e.clientY;
                        startScrollTop = wheel.scrollTop;
                        wheel.classList.add("dragging");
                        try {{ wheel.setPointerCapture(e.pointerId); }} catch (_err) {{}}
                    }};
                    const onPointerMove = (e) => {{
                        if (!isDragging) return;
                        const dy = e.clientY - startY;
                        wheel.scrollTop = startScrollTop - dy;
                    }};
                    const onPointerUp = (e) => {{
                        if (!isDragging) return;
                        isDragging = false;
                        wheel.classList.remove("dragging");
                        try {{ wheel.releasePointerCapture(e.pointerId); }} catch (_err) {{}}
                        snapToNearest();
                    }};
                    wheel.addEventListener("pointerdown", onPointerDown);
                    wheel.addEventListener("pointermove", onPointerMove);
                    wheel.addEventListener("pointerup", onPointerUp);
                    wheel.addEventListener("pointercancel", onPointerUp);
                }}

                makeItems(wm, months.map((_, i) => i + 1), v => months[v - 1]);
                makeItems(wd, days, v => String(v));
                makeItems(wy, years, v => String(v));
                scrollToValue(wm, months.map((_, i) => i + 1), month);
                scrollToValue(wd, days, day);
                scrollToValue(wy, years, year);

                bind(wm, months.map((_, i) => i + 1), v => {{ month = v; }});
                bind(wd, days, v => {{ day = v; }});
                bind(wy, years, v => {{ year = v; }});

                refresh();
            </script>
        </body>
        </html>
        """
        components.html(age_wheel_component_html, height=360, scrolling=False)
        st.markdown('<div class="weight-next-btn">', unsafe_allow_html=True)
        if st.button("Next", use_container_width=True, key="age_next_btn"):
            today = date.today()
            by = int(st.session_state.get("onboarding_birth_year", today.year - int(defaults["age"])))
            bm = int(st.session_state.get("onboarding_birth_month", 1))
            bd = int(st.session_state.get("onboarding_birth_day", 1))
            safe_bd = bd
            while safe_bd > 28:
                try:
                    _ = date(by, bm, safe_bd)
                    break
                except ValueError:
                    safe_bd -= 1
            dob = date(by, bm, safe_bd)
            age_years = int((today - dob).days // 365.2425)
            age_years = max(16, min(75, age_years))
            st.session_state["onboarding_birth_day"] = safe_bd
            st.session_state["onboarding_age"] = age_years
            st.session_state["onboarding_stage"] = "goal_weight"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-card
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-page-shell
        return None

    if st.session_state["onboarding_stage"] == "goal_weight":
        current_weight = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
        current_goal_weight = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
        current_height = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
        selected_sex_for_goal_link = st.session_state.get("onboarding_selected_sex", "M")
        delta_pct = ((current_weight - current_goal_weight) / max(current_weight, 1.0)) * 100.0

        st.markdown('<div class="goal-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="weight-page-shell">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="height-top-back-wrap"><a id="goal-weight-back-link" class="height-top-back-link" href="?onboarding_sex={selected_sex_for_goal_link}&onboarding_height={current_height}&onboarding_weight={current_weight:.1f}&onboarding_goal_weight={current_goal_weight:.1f}&onboarding_goal_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weight-card">', unsafe_allow_html=True)
        st.markdown(
            """
                <div class="assessment-card" style="margin-top:0; margin-bottom:10px; box-shadow:none; padding:1.15rem 0.2rem 0.8rem 0.2rem;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:60%;"></div></div>
                    <div class="assessment-question">What is your goal weight?</div>
                    <div class="assessment-helper">Set a clear target and we will tailor your plan around it.</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        goal_ruler_component_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                }}
                .stage {{
                    min-height: 190px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                }}
                .live-value {{
                    text-align: center;
                    font-size: 2.2rem;
                    line-height: 1;
                    font-weight: 900;
                    color: #26314d;
                    margin: 0.1rem 0 0.5rem 0;
                }}
                .live-value small {{
                    font-size: 0.52em;
                    font-weight: 750;
                    color: #6b7280;
                    margin-left: 5px;
                }}
                .live-goal {{
                    margin-top: 0.6rem;
                    border: 1px solid rgba(31, 41, 55, 0.1);
                    border-radius: 12px;
                    padding: 0.45rem 0.7rem 0.55rem 0.7rem;
                    background: #fbfcfc;
                    width: min(92vw, 760px);
                    text-align: center;
                    color: #1f2937;
                    font-size: 1.02rem;
                    font-weight: 800;
                }}
                .ruler {{
                    position: relative;
                    width: min(92vw, 760px);
                    height: 86px;
                    border-radius: 12px;
                    background: rgba(235, 251, 244, 0.95);
                    overflow: hidden;
                    cursor: ew-resize;
                    user-select: none;
                    touch-action: none;
                }}
                .track {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 100%;
                    transform: translateX(0);
                    will-change: transform;
                }}
                .tick {{
                    position: absolute;
                    bottom: 22px;
                    width: 1px;
                    background: rgba(37, 164, 124, 0.36);
                    height: 18px;
                }}
                .tick.major {{
                    height: 30px;
                    background: rgba(37, 164, 124, 0.58);
                }}
                .tick .tick-label {{
                    position: absolute;
                    top: 34px;
                    left: -12px;
                    min-width: 24px;
                    text-align: center;
                    color: rgba(66, 82, 110, 0.92);
                    font-size: 0.98rem;
                    font-weight: 850;
                    line-height: 1;
                }}
                .pointer {{
                    position: absolute;
                    top: 12px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 10px;
                    height: 34px;
                    border-radius: 999px;
                    background: #27d978;
                    pointer-events: none;
                }}
                .next-wrap {{
                    margin-top: 14px;
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }}
                .next-btn {{
                    width: min(92%, 720px);
                    border: none;
                    border-radius: 999px;
                    background: #25d366;
                    color: #ffffff;
                    font-size: 1.95rem;
                    font-weight: 900;
                    min-height: 56px;
                    letter-spacing: 0.01em;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                }}
            </style>
        </head>
        <body>
            <div class="stage">
                <div id="liveGoal" class="live-value"></div>
                <div id="ruler" class="ruler">
                    <div id="track" class="track"></div>
                    <div class="pointer"></div>
                </div>
                <div id="liveGoalMeta" class="live-goal"></div>
            </div>
            <script>
                const minW = 35.0, maxW = 250.0;
                const majorEvery = 5.0;
                const currentWeight = {current_weight:.2f};
                const ruler = document.getElementById("ruler");
                const track = document.getElementById("track");
                const liveGoal = document.getElementById("liveGoal");
                const liveGoalMeta = document.getElementById("liveGoalMeta");
                let current = {current_goal_weight:.1f};
                let dragging = false;
                let dragStartX = 0;
                let dragStartWeight = current;
                let lastCommitted = current;

                const pxPerKg = 22;
                const trackPad = 460;
                const rulerWidth = () => ruler.clientWidth || 760;
                const centerX = () => rulerWidth() / 2;
                const trackWidth = () => trackPad * 2 + (maxW - minW) * pxPerKg;

                function clamp(v, lo, hi) {{
                    return Math.max(lo, Math.min(hi, v));
                }}
                function round1(v) {{
                    return Math.round(v * 10) / 10;
                }}
                function xForWeight(w) {{
                    return trackPad + (w - minW) * pxPerKg;
                }}
                function buildTrack() {{
                    track.style.width = `${{Math.round(trackWidth())}}px`;
                    const parts = [];
                    for (let w = minW; w <= maxW + 0.0001; w += 0.5) {{
                        const ww = Math.round(w * 10) / 10;
                        const x = xForWeight(ww);
                        const isMajor = Math.abs((ww / majorEvery) - Math.round(ww / majorEvery)) < 0.0001;
                        const label = isMajor ? `<span class="tick-label">${{Math.round(ww)}}</span>` : "";
                        parts.push(`<div class="tick ${{isMajor ? "major" : "minor"}}" style="left:${{Math.round(x)}}px">${{label}}</div>`);
                    }}
                    track.innerHTML = parts.join("");
                }}
                function render(w) {{
                    current = clamp(round1(w), minW, maxW);
                    const tx = centerX() - xForWeight(current);
                    track.style.transform = `translateX(${{Math.round(tx)}}px)`;
                    const pct = ((currentWeight - current) / Math.max(currentWeight, 1)) * 100;
                    const mode = pct >= 0 ? "lose" : "gain";
                    if (liveGoal) liveGoal.innerHTML = `${{current.toFixed(1)}}<small>kg</small>`;
                    if (liveGoalMeta) liveGoalMeta.textContent = `Goal change: ${{Math.abs(pct).toFixed(1)}}% ${{mode}}`;
                    try {{
                        window.parent.postMessage({{
                            type: "onboarding-goal-weight",
                            value: current,
                            sex: "{selected_sex_for_goal_link}",
                        }}, "*");
                    }} catch (e) {{
                        // Ignore bridge failures.
                    }}
                }}
                function commitGoalWeightToApp() {{
                    // Avoid top-level URL mutations from inside iframe.
                    // They can trigger Streamlit removeChild errors on some browsers.
                    return;
                }}
                function onMoveClientX(clientX) {{
                    if (!dragging) return;
                    const deltaKg = (clientX - dragStartX) / pxPerKg;
                    render(dragStartWeight - deltaKg);
                }}
                function startDrag(clientX) {{
                    dragging = true;
                    dragStartX = clientX;
                    dragStartWeight = current;
                }}
                function endDrag(clientX = null) {{
                    if (!dragging) return;
                    if (clientX !== null) {{
                        const deltaKg = (clientX - dragStartX) / pxPerKg;
                        render(dragStartWeight - deltaKg);
                    }}
                    dragging = false;
                    if (Math.abs(current - lastCommitted) >= 0.1) {{
                        lastCommitted = current;
                        commitGoalWeightToApp();
                    }}
                }}
                ruler.addEventListener("pointerdown", (e) => startDrag(e.clientX));
                window.addEventListener("pointermove", (e) => onMoveClientX(e.clientX));
                window.addEventListener("pointerup", (e) => endDrag(e.clientX));
                window.addEventListener("pointercancel", () => endDrag());
                ruler.addEventListener("mousedown", (e) => {{ e.preventDefault(); startDrag(e.clientX); }});
                window.addEventListener("mousemove", (e) => onMoveClientX(e.clientX));
                window.addEventListener("mouseup", (e) => endDrag(e.clientX));
                ruler.addEventListener("touchstart", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    startDrag(e.touches[0].clientX);
                }}, {{ passive: true }});
                window.addEventListener("touchmove", (e) => {{
                    if (!e.touches || !e.touches.length) return;
                    onMoveClientX(e.touches[0].clientX);
                }}, {{ passive: true }});
                window.addEventListener("touchend", (e) => {{
                    if (!e.changedTouches || !e.changedTouches.length) {{ endDrag(); return; }}
                    endDrag(e.changedTouches[0].clientX);
                }}, {{ passive: true }});
                window.addEventListener("wheel", (e) => {{ e.preventDefault(); }}, {{ passive: false }});
                window.addEventListener("resize", () => render(current));
                buildTrack();
                render(current);
            </script>
        </body>
        </html>
        """
        components.html(goal_ruler_component_html, height=220, scrolling=False)
        st.markdown(
            f'<div class="weight-next-btn"><a id="goal-weight-next-link" class="height-next-link-native" href="?onboarding_sex={selected_sex_for_goal_link}&onboarding_height={current_height}&onboarding_weight={current_weight:.1f}&onboarding_goal_weight={current_goal_weight:.1f}&onboarding_goal_weight_next=1" target="_self" rel="noopener">Next</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-card
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-page-shell
        return None

    if st.session_state["onboarding_stage"] == "goal_timeline":
        today = date.today()
        current_weight = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
        goal_weight = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
        current_height = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
        selected_sex_for_goal_time_back = st.session_state.get("onboarding_selected_sex", "M")

        default_goal_year = int(st.session_state.get("onboarding_goal_year", today.year))
        default_goal_month = int(st.session_state.get("onboarding_goal_month", today.month))
        default_goal_day = int(st.session_state.get("onboarding_goal_day", today.day))
        safe_goal_day = default_goal_day
        while safe_goal_day > 28:
            try:
                _ = date(default_goal_year, default_goal_month, safe_goal_day)
                break
            except ValueError:
                safe_goal_day -= 1
        st.session_state["onboarding_goal_day"] = safe_goal_day
        goal_date = date(default_goal_year, default_goal_month, safe_goal_day)
        days_until = max(7, (goal_date - today).days)
        weeks_until = max(1, int(round(days_until / 7.0)))
        weekly_delta = abs(current_weight - goal_weight) / max(weeks_until, 1)

        st.markdown('<div class="goal-time-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="weight-page-shell">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_sex={selected_sex_for_goal_time_back}&onboarding_height={current_height}&onboarding_weight={current_weight:.1f}&onboarding_goal_weight={goal_weight:.1f}&onboarding_goal_time_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weight-card">', unsafe_allow_html=True)
        st.markdown(
            """
                <div class="assessment-card" style="margin-top:0; margin-bottom:10px; box-shadow:none; padding:1.15rem 0.2rem 0.8rem 0.2rem;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:72%;"></div></div>
                    <div class="assessment-question">When do you want to reach your goal weight?</div>
                    <div class="assessment-helper">Pick a target date and we will pace your weekly plan.</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        goal_time_component_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                }}
                .wrap {{
                    width: min(94vw, 760px);
                    margin: 0 auto;
                }}
                .wheels {{
                    display: grid;
                    grid-template-columns: 1.65fr 0.9fr 1fr;
                    gap: 10px;
                    align-items: center;
                    margin-top: 4px;
                }}
                .wheel {{
                    height: 210px;
                    overflow-y: auto;
                    scroll-snap-type: y mandatory;
                    scroll-behavior: smooth;
                    -webkit-overflow-scrolling: touch;
                    scrollbar-width: none;
                    -ms-overflow-style: none;
                    background: #f8fafb;
                    border-radius: 14px;
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    position: relative;
                    cursor: grab;
                }}
                .wheel.dragging {{
                    cursor: grabbing;
                }}
                .wheel::-webkit-scrollbar {{ display: none; }}
                .item {{
                    height: 54px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.18rem;
                    font-weight: 700;
                    color: rgba(15, 23, 42, 0.34);
                    scroll-snap-align: center;
                    user-select: none;
                    position: relative;
                    z-index: 3;
                    transition: color 120ms ease, font-size 120ms ease, font-weight 120ms ease;
                }}
                .item.active {{
                    color: #1f2937;
                    font-size: 2rem;
                    font-weight: 900;
                    background: rgba(15, 23, 42, 0.06);
                    border-radius: 10px;
                }}
                .goal-time-card {{
                    margin-top: 26px;
                    border: 1px solid rgba(34, 197, 94, 0.22);
                    border-radius: 16px;
                    background: #fbfcfc;
                    padding: 12px 14px;
                }}
                .goal-time-top {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    align-items: center;
                }}
                .goal-time-kpi {{
                    display: flex;
                    align-items: baseline;
                    gap: 8px;
                    color: #1f2937;
                    font-size: 1.15rem;
                    font-weight: 800;
                }}
                .goal-time-kpi .badge {{
                    min-width: 58px;
                    text-align: center;
                    background: #dcfce7;
                    color: #16a34a;
                    font-size: 2rem;
                    font-weight: 900;
                    line-height: 1;
                    border-radius: 12px;
                    padding: 5px 10px;
                }}
                .goal-time-note {{
                    margin-top: 10px;
                    padding-top: 10px;
                    border-top: 1px solid rgba(148, 163, 184, 0.2);
                    color: #4b5563;
                    font-size: 1rem;
                    line-height: 1.42;
                    font-weight: 700;
                }}
            </style>
        </head>
        <body>
            <div class="wrap">
                <div class="wheels">
                    <div id="wheel-goal-month" class="wheel"></div>
                    <div id="wheel-goal-day" class="wheel"></div>
                    <div id="wheel-goal-year" class="wheel"></div>
                </div>
                <div class="goal-time-card">
                    <div class="goal-time-top">
                        <div class="goal-time-kpi"><span>Need about</span><span id="goal-weeks-live" class="badge">{weeks_until}</span><span>weeks</span></div>
                        <div class="goal-time-kpi"><span>Weekly</span><span id="goal-rate-live" class="badge">{weekly_delta:.2f}</span><span>kg</span></div>
                    </div>
                </div>
            </div>
            <script>
                const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
                const days = Array.from({{length: 31}}, (_, i) => i + 1);
                const currentYear = new Date().getFullYear();
                const years = [];
                for (let y = currentYear; y <= currentYear + 5; y++) years.push(y);
                const currentWeight = {current_weight:.2f};
                const goalWeight = {goal_weight:.2f};

                let month = {default_goal_month};
                let day = {safe_goal_day};
                let year = {default_goal_year};

                const wm = document.getElementById("wheel-goal-month");
                const wd = document.getElementById("wheel-goal-day");
                const wy = document.getElementById("wheel-goal-year");
                const weeksLive = document.getElementById("goal-weeks-live");
                const rateLive = document.getElementById("goal-rate-live");

                function makeItems(wheel, values, labeler) {{
                    wheel.innerHTML = "";
                    for (let i = 0; i < 2; i++) {{
                        const pad = document.createElement("div");
                        pad.className = "item";
                        pad.textContent = "";
                        wheel.appendChild(pad);
                    }}
                    values.forEach(v => {{
                        const d = document.createElement("div");
                        d.className = "item";
                        d.dataset.value = String(v);
                        d.textContent = labeler(v);
                        wheel.appendChild(d);
                    }});
                    for (let i = 0; i < 2; i++) {{
                        const pad = document.createElement("div");
                        pad.className = "item";
                        pad.textContent = "";
                        wheel.appendChild(pad);
                    }}
                }}
                function scrollToValue(wheel, values, value) {{
                    const idx = values.indexOf(value);
                    const row = 54;
                    const target = Math.max(0, (idx + 2) * row - (wheel.clientHeight / 2) + (row / 2));
                    wheel.scrollTop = target;
                }}
                function markActive(wheel, selected) {{
                    wheel.querySelectorAll(".item").forEach(el => {{
                        const v = el.dataset.value;
                        el.classList.toggle("active", v !== undefined && String(selected) === v);
                    }});
                }}
                function daysInMonth(y, m) {{
                    return new Date(y, m, 0).getDate();
                }}
                function safeDate(y, m, d) {{
                    const md = daysInMonth(y, m);
                    return new Date(y, m - 1, Math.min(d, md));
                }}
                function refresh() {{
                    const maxDay = daysInMonth(year, month);
                    if (day > maxDay) day = maxDay;

                    markActive(wm, month);
                    markActive(wd, day);
                    markActive(wy, year);

                    const today = new Date();
                    today.setHours(0, 0, 0, 0);
                    const target = safeDate(year, month, day);
                    target.setHours(0, 0, 0, 0);
                    const diffDays = Math.max(7, Math.ceil((target - today) / 86400000));
                    const weeks = Math.max(1, Math.round(diffDays / 7));
                    const weekly = Math.abs(currentWeight - goalWeight) / Math.max(weeks, 1);

                    weeksLive.textContent = String(weeks);
                    rateLive.textContent = weekly.toFixed(2);
                }}
                function bind(wheel, values, onPick) {{
                    let t = null;
                    const row = 54;
                    const snapToNearest = () => {{
                        const center = wheel.scrollTop + wheel.clientHeight / 2;
                        const idx = Math.round((center - row / 2) / row) - 2;
                        const clamped = Math.max(0, Math.min(values.length - 1, idx));
                        const target = Math.max(0, (clamped + 2) * row - (wheel.clientHeight / 2) + (row / 2));
                        wheel.scrollTo({{ top: target, behavior: "smooth" }});
                        onPick(values[clamped]);
                        refresh();
                    }};
                    const onScroll = () => {{
                        if (t) window.clearTimeout(t);
                        t = window.setTimeout(() => {{
                            snapToNearest();
                        }}, 90);
                    }};
                    wheel.addEventListener("scroll", onScroll, {{ passive: true }});

                    let isDragging = false;
                    let startY = 0;
                    let startScrollTop = 0;
                    const onPointerDown = (e) => {{
                        isDragging = true;
                        startY = e.clientY;
                        startScrollTop = wheel.scrollTop;
                        wheel.classList.add("dragging");
                        try {{ wheel.setPointerCapture(e.pointerId); }} catch (_err) {{}}
                    }};
                    const onPointerMove = (e) => {{
                        if (!isDragging) return;
                        const dy = e.clientY - startY;
                        wheel.scrollTop = startScrollTop - dy;
                    }};
                    const onPointerUp = (e) => {{
                        if (!isDragging) return;
                        isDragging = false;
                        wheel.classList.remove("dragging");
                        try {{ wheel.releasePointerCapture(e.pointerId); }} catch (_err) {{}}
                        snapToNearest();
                    }};
                    wheel.addEventListener("pointerdown", onPointerDown);
                    wheel.addEventListener("pointermove", onPointerMove);
                    wheel.addEventListener("pointerup", onPointerUp);
                    wheel.addEventListener("pointercancel", onPointerUp);
                }}

                makeItems(wm, months.map((_, i) => i + 1), v => months[v - 1]);
                makeItems(wd, days, v => String(v));
                makeItems(wy, years, v => String(v));
                scrollToValue(wm, months.map((_, i) => i + 1), month);
                scrollToValue(wd, days, day);
                scrollToValue(wy, years, year);
                bind(wm, months.map((_, i) => i + 1), v => {{ month = v; }});
                bind(wd, days, v => {{ day = v; }});
                bind(wy, years, v => {{ year = v; }});
                refresh();
            </script>
        </body>
        </html>
        """
        components.html(goal_time_component_html, height=340, scrolling=False)
        st.markdown('<div class="weight-next-btn">', unsafe_allow_html=True)
        if st.button("Next", use_container_width=True, key="goal_time_next_btn"):
            goal_y = int(st.session_state.get("onboarding_goal_year", default_goal_year))
            goal_m = int(st.session_state.get("onboarding_goal_month", default_goal_month))
            goal_d = int(st.session_state.get("onboarding_goal_day", safe_goal_day))
            safe_goal_d = goal_d
            while safe_goal_d > 28:
                try:
                    _ = date(goal_y, goal_m, safe_goal_d)
                    break
                except ValueError:
                    safe_goal_d -= 1
            target_dt = date(goal_y, goal_m, safe_goal_d)
            day_gap = max(7, (target_dt - today).days)
            st.session_state["onboarding_goal_day"] = safe_goal_d
            st.session_state["onboarding_goal_weeks"] = max(1, int(round(day_gap / 7.0)))
            st.session_state["onboarding_weekly_delta_kg"] = abs(current_weight - goal_weight) / max(
                st.session_state["onboarding_goal_weeks"], 1
            )
            st.session_state["onboarding_stage"] = "plan_intro"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-card
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-page-shell
        return None

    if st.session_state["onboarding_stage"] == "plan_intro":
        current_weight_for_back = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
        current_goal_weight_for_back = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
        current_height_for_back = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
        selected_sex_for_back = st.session_state.get("onboarding_selected_sex", "M")
        st.markdown('<div class="plan-intro-stage-lock"></div>', unsafe_allow_html=True)
        st.markdown('<div class="weight-page-shell">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_sex={selected_sex_for_back}&onboarding_height={current_height_for_back}&onboarding_weight={current_weight_for_back:.1f}&onboarding_goal_weight={current_goal_weight_for_back:.1f}&onboarding_plan_intro_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="max-width:820px; margin: 0 auto;">
                <div class="assessment-card" style="margin-top:0; margin-bottom:14px; box-shadow:none; padding:1.2rem 1.1rem 1.05rem 1.1rem;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:84%;"></div></div>
                    <div style="margin-top:24px; display:flex; justify-content:center;">
                        <div style="
                            width:min(95%, 700px);
                            min-height:215px;
                            border-radius:18px;
                            border:2px solid rgba(34, 197, 94, 0.7);
                            background:rgba(255,255,255,0.55);
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            position:relative;
                            padding:26px 22px;">
                            <div style="
                                position:absolute; top:-24px; left:50%; transform:translateX(-50%);
                                width:48px; height:48px; border-radius:999px;
                                background:#4ade80; box-shadow:0 8px 18px rgba(34,197,94,0.35);
                                display:flex; align-items:center; justify-content:center;
                                color:#ffffff; font-size:1.45rem;">💡</div>
                            <div style="
                                color:#374151;
                                text-align:center;
                                font-size:2.35rem;
                                line-height:1.35;
                                font-weight:900;">
                                Next, we will ask a few questions<br/>to estimate the best diet plan for you
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weight-next-btn">', unsafe_allow_html=True)
        if st.button("Next", use_container_width=True, key="plan_intro_next_btn"):
            st.session_state["onboarding_stage"] = "details"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # .weight-page-shell
        return None

    current_weight_for_back = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
    current_goal_weight_for_back = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
    current_height_for_back = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
    selected_sex_for_back = st.session_state.get("onboarding_selected_sex", "M")
    st.markdown('<div class="details-stage-lock"></div>', unsafe_allow_html=True)
    st.markdown('<div class="basic-info-wrap">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_sex={selected_sex_for_back}&onboarding_height={current_height_for_back}&onboarding_weight={current_weight_for_back:.1f}&onboarding_goal_weight={current_goal_weight_for_back:.1f}&onboarding_details_back=1" target="_self" rel="noopener">Back</a></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
            <div class="assessment-card" style="margin-top:0; margin-bottom:12px; box-shadow:none; padding:1.15rem 1rem 0.95rem 1rem;">
                <div class="assessment-progress"><div class="assessment-progress-fill" style="width:96%;"></div></div>
                <div class="assessment-question" style="font-size:2.1rem; margin-top:0.55rem;">Lifestyle Questions</div>
                <div class="assessment-helper">Help us personalize your daily nutrition and training routine.</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    schedule_options = ["Regular daytime", "Night shift", "Irregular schedule"]
    workout_location_options = ["Gym", "Home", "Both"]
    workout_time_options = ["15-20 minutes", "30-45 minutes", "60+ minutes"]
    diet_preference_options = ["No preference", "Vegetarian", "High protein", "Low carb"]
    craving_options = ["Rarely", "Sometimes", "Often"]
    stress_options = ["Low", "Medium", "High"]
    sleep_options = ["Good", "Average", "Poor"]
    health_options = [
        "None",
        "Knee pain",
        "Back pain",
        "Shoulder injury",
        "Joint pain",
        "Type 2 Diabetes",
        "Dyslipidemia (High blood lipids)",
        "PCOS",
        "Fatty Liver",
        "Coronary Heart Disease",
        "Chronic Kidney Disease",
        "Sleep Apnea",
        "Severe Arthritis",
        "Hypertension",
        "High Uric Acid",
        "Hypothyroidism",
    ]

    def _safe_index(options: list[str], value: str) -> int:
        return options.index(value) if value in options else 0

    def _normalize_health_conditions(selected: list[str] | None) -> list[str]:
        values = [v for v in (selected or []) if v in health_options]
        if not values:
            return ["None"]
        if "None" in values and len(values) > 1:
            values = [v for v in values if v != "None"]
        return values or ["None"]

    with st.form("basic_information_form"):
        schedule_type = st.radio(
            "What is your daily schedule like?",
            schedule_options,
            index=_safe_index(schedule_options, str(defaults.get("schedule_type", "Regular daytime"))),
            horizontal=True,
        )
        workout_location = st.radio(
            "Where do you prefer to work out?",
            workout_location_options,
            index=_safe_index(workout_location_options, str(defaults.get("workout_location", "Home"))),
            horizontal=True,
        )
        workout_time = st.radio(
            "How much time can you exercise per session?",
            workout_time_options,
            index=_safe_index(workout_time_options, str(defaults.get("workout_time", "30-45 minutes"))),
            horizontal=True,
        )
        diet_preference = st.radio(
            "Diet preference",
            diet_preference_options,
            index=_safe_index(diet_preference_options, str(defaults.get("diet_preference", "No preference"))),
            horizontal=True,
        )
        craving_level = st.radio(
            "Do you often crave sweets/snacks?",
            craving_options,
            index=_safe_index(craving_options, str(defaults.get("craving_level", "Sometimes"))),
            horizontal=True,
        )
        stress_level = st.radio(
            "Stress level",
            stress_options,
            index=_safe_index(stress_options, str(defaults.get("stress_level", "Medium"))),
            horizontal=True,
        )
        sleep_quality = st.radio(
            "Sleep quality",
            sleep_options,
            index=_safe_index(sleep_options, str(defaults.get("sleep_quality", "Average"))),
            horizontal=True,
        )
        default_conditions = defaults.get("health_conditions", ["None"])
        if isinstance(default_conditions, str):
            default_conditions = [x.strip() for x in default_conditions.split(",") if x.strip()]
        default_conditions = [c for c in default_conditions if c in health_options]
        if not default_conditions:
            default_conditions = ["None"]
        st.markdown("**Do you have any injuries or health conditions?**")
        picked_conditions: list[str] = []
        pills_per_row = 4
        for row_start in range(0, len(health_options), pills_per_row):
            row_options = health_options[row_start : row_start + pills_per_row]
            row_cols = st.columns(len(row_options), gap="small")
            for idx, (col, option) in enumerate(zip(row_cols, row_options)):
                option_key = f"onboarding_health_condition_{row_start + idx}"
                with col:
                    is_selected = st.checkbox(
                        option,
                        value=option in default_conditions,
                        key=option_key,
                    )
                if is_selected:
                    picked_conditions.append(option)
        submitted = st.form_submit_button("Next", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return None

    sex = selected_sex if selected_sex in {"M", "F"} else defaults["sex"]
    age = int(st.session_state.get("onboarding_age", defaults["age"]))
    height_cm = float(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))
    weight_kg = float(st.session_state.get("onboarding_weight_kg", defaults["weight_kg"]))
    goal_weight_kg = float(st.session_state.get("onboarding_goal_weight_kg", defaults["goal_weight_kg"]))
    goal_timeline_weeks = int(st.session_state.get("onboarding_goal_weeks", 12))
    workout_time_to_days = {"15-20 minutes": 3, "30-45 minutes": 4, "60+ minutes": 5}
    days_per_week = int(workout_time_to_days.get(workout_time, defaults["days_per_week"]))
    health_conditions = _normalize_health_conditions(picked_conditions)
    lifestyle_text = (
        f"{schedule_type.lower()} schedule. "
        f"Prefers {workout_location.lower()} workouts and {workout_time} sessions. "
        f"Diet preference is {diet_preference.lower()}. "
        f"Cravings: {craving_level.lower()}, stress: {stress_level.lower()}, sleep: {sleep_quality.lower()}. "
        f"Health conditions: {', '.join(health_conditions)}."
    )
    st.session_state["onboarding_stage"] = "height"
    st.session_state.pop("onboarding_selected_sex", None)
    st.session_state.pop("onboarding_weight_kg", None)
    return {
        "name": str(defaults.get("name", "")).strip(),
        "age": int(age),
        "sex": sex,
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "goal_weight_kg": float(goal_weight_kg),
        "goal_timeline_weeks": int(goal_timeline_weeks),
        "days_per_week": int(days_per_week),
        "lifestyle_text": lifestyle_text,
        "schedule_type": schedule_type,
        "workout_location": workout_location,
        "workout_time": workout_time,
        "diet_preference": diet_preference,
        "craving_level": craving_level,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "health_conditions": health_conditions,
    }



def render_plan_header(predicted_body_type: str, bmi: float, calorie_target: int, similarity: float, weight_kg: float, goal_weight_kg: float, days_per_week: int) -> None:
    st.markdown(
        f"""
        <div class="hero-card hero-dark">
            <div class="hero-title">Your Personalized Plan</div>
            <div class="hero-sub">Lifestyle-weighted guidance based on your profile and routine constraints.</div>
            <span class="chip">Body Type: {predicted_body_type}</span>
            <span class="chip">BMI: {bmi:.1f}</span>
            <span class="chip">Daily Calories: {calorie_target} kcal</span>
            <span class="chip">Diet Twin Match: {similarity:.1%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    c1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Current Weight</div>
            <div class="metric-value">{weight_kg:.1f} kg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Goal Weight</div>
            <div class="metric-value">{goal_weight_kg:.1f} kg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Workout Days</div>
            <div class="metric-value">{days_per_week} / week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_twin_section(twin: pd.Series) -> None:
    profile_id = html.escape(str(twin.get("profile_id", "")))
    diet_pattern = html.escape(str(twin.get("diet_pattern", "")))
    adherence = float(twin.get("adherence_score", 0))
    notes = html.escape(str(twin.get("notes", "")))
    st.markdown(
        f"""
        <div class="twin-section-box">
            <div class="section-title">Your Closest Diet Twin</div>
            <div class="small-note" style="font-size:1rem; margin-bottom:0.6rem;">
                Matched profile: <b>{profile_id}</b> | Pattern: <b>{diet_pattern}</b> | Avg adherence: <b>{adherence:.0f}%</b>
            </div>
            <div class="block-card">
                <div>{notes}</div>
                <div class="small-note" style="margin-top:6px;">
                    Lifestyle-weighted matching is prioritized so recommendations reflect your daily reality, not just body stats.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mint_table(df: pd.DataFrame) -> None:
    table_html = df.to_html(index=False, classes="mint-table", border=0, escape=True)
    st.markdown(f'<div class="mint-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def render_diet_plan_tab(meals: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Recommended Meal Structure</div>', unsafe_allow_html=True)
    meal_table = meals[["meal_type", "food_name", "calories", "protein_g", "carbs_g", "fat_g"]].copy()
    meal_table.columns = ["Meal", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
    _render_mint_table(meal_table)
    total_protein = meals["protein_g"].sum() if not meals.empty else 0.0
    st.markdown(
        f"""
        <div class="small-note">
            Meals are selected by calorie fit and protein priority.
            Estimated daily protein from this meal set: <b>{total_protein:.0f} g</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lifestyle_fit_tab(
    lifestyle: dict,
    lifestyle_fit: dict | None = None,
    recommendations: list[str] | None = None,
) -> None:
    st.markdown('<div class="section-title">Why This Plan Fits Your Lifestyle</div>', unsafe_allow_html=True)
    if lifestyle_fit:
        f1, f2, f3 = st.columns(3)
        f1.metric("Lifestyle Fit", f"{lifestyle_fit['score']:.0f} / 100", lifestyle_fit["label"])
        f2.metric("Predicted Pattern", lifestyle_fit["pattern"])
        f3.metric("Model Confidence", f"{lifestyle_fit['pattern_confidence']:.0%}")

        st.markdown(
            f"""
            <div class="block-card">
                <b>Machine learning output:</b> A Random Forest model predicts that this plan has a
                <b>{str(lifestyle_fit['label']).lower()}</b> for your current routine. The raw model estimate was
                <b>{lifestyle_fit['raw_score']:.0f}%</b>, then the app applied small safety adjustments for sleep,
                injuries, and medical constraints.
            </div>
            """,
            unsafe_allow_html=True,
        )

    cues = lifestyle.get("matched_cues", [])
    if cues:
        st.markdown("- " + "\n- ".join([f"We adjusted recommendations because {c}." for c in cues[:5]]))
    else:
        st.markdown(
            "- We used your body profile and goal to personalize the baseline plan.\n"
            "- Add more daily routine details in the lifestyle box for deeper personalization."
        )
    if lifestyle.get("injury_care") or lifestyle.get("low_sleep"):
        st.info("Recovery-sensitive mode is on: prioritize consistency and moderate intensity.")

    if recommendations:
        st.markdown("### Fit Recommendations")
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

    if lifestyle_fit:
        st.markdown("### Strongest Model Drivers")
        driver_labels = {
            "age": "Age",
            "height_cm": "Height",
            "weight_kg": "Weight",
            "sex_bin": "Sex",
            "night_shift": "Night shift schedule",
            "sugar_craving": "Sweet/snack cravings",
            "home_workout": "Home workout preference",
            "vegetarian_pref": "Vegetarian preference",
            "high_stress": "High stress",
            "short_sessions": "Short workout sessions",
            "goal_direction": "Goal direction",
        }
        driver_df = lifestyle_fit["top_features"].reset_index()
        driver_df.columns = ["Feature", "Importance"]
        driver_df["Feature"] = driver_df["Feature"].map(driver_labels).fillna(driver_df["Feature"])
        driver_df["Importance"] = (driver_df["Importance"] * 100).round(1)
        _render_mint_table(driver_df)

        with st.expander("How the lifestyle model works"):
            st.markdown(
                f"""
                The app trains a `RandomForestRegressor` on `data/diet_lifestyle_profiles.csv`.
                Inputs are age, height, weight, sex, goal direction, and lifestyle flags such as
                night shift, cravings, home workouts, vegetarian preference, high stress, and short sessions.

                The target is historical `adherence_score`, so the output is an estimated likelihood that the
                generated plan matches the user's routine. A companion `RandomForestClassifier` predicts the
                closest diet pattern label. On the current holdout split, the adherence model has MAE
                `{lifestyle_fit['metrics']['mae']:.1f}` points and R2 `{lifestyle_fit['metrics']['r2']:.2f}`.
                """
            )


def render_profile_form_ui(defaults: dict) -> dict | None:
    schedule_options = ["Regular daytime", "Night shift", "Irregular schedule"]
    workout_location_options = ["Gym", "Home", "Both"]
    workout_time_options = ["15-20 minutes", "30-45 minutes", "60+ minutes"]
    diet_preference_options = ["No preference", "Vegetarian", "High protein", "Low carb"]
    craving_options = ["Rarely", "Sometimes", "Often"]
    stress_options = ["Low", "Medium", "High"]
    sleep_options = ["Good", "Average", "Poor"]
    health_options = [
        "None",
        "Knee pain",
        "Back pain",
        "Shoulder injury",
        "Joint pain",
        "Type 2 Diabetes",
        "Dyslipidemia (High blood lipids)",
        "PCOS",
        "Fatty Liver",
        "Coronary Heart Disease",
        "Chronic Kidney Disease",
        "Sleep Apnea",
        "Severe Arthritis",
        "Hypertension",
        "High Uric Acid",
        "Hypothyroidism",
    ]

    def _safe_index(options: list[str], value: str) -> int:
        return options.index(value) if value in options else 0

    def _normalize_health_conditions(selected: list[str] | None) -> list[str]:
        values = [v for v in (selected or []) if v in health_options]
        if not values:
            return ["None"]
        if "None" in values and len(values) > 1:
            values = [v for v in values if v != "None"]
        return values or ["None"]

    with st.form("profile_form"):
        st.subheader("Tell us about you")
        age = st.slider("Age", 16, 75, int(defaults.get("age", 24)))
        sex = st.selectbox("Sex", ["M", "F"], index=0 if defaults.get("sex", "M") == "M" else 1)
        height_cm = st.slider("Height (cm)", 145, 210, int(float(defaults.get("height_cm", 172))))
        weight_kg = st.slider("Current weight (kg)", 40, 160, int(float(defaults.get("weight_kg", 75))))
        goal_weight_kg = st.slider("Goal weight (kg)", 40, 160, int(float(defaults.get("goal_weight_kg", 68))))
        goal_timeline_weeks = st.slider(
            "How many weeks do you expect to reach your goal?", 4, 52, int(defaults.get("goal_timeline_weeks", 12))
        )
        days_per_week = st.slider("Workout days / week", 0, 7, int(defaults.get("days_per_week", 4)))
        schedule_type = st.selectbox(
            "What is your daily schedule like?",
            schedule_options,
            index=_safe_index(schedule_options, str(defaults.get("schedule_type", "Regular daytime"))),
        )
        workout_location = st.selectbox(
            "Where do you prefer to work out?",
            workout_location_options,
            index=_safe_index(workout_location_options, str(defaults.get("workout_location", "Home"))),
        )
        workout_time = st.selectbox(
            "How much time can you exercise per session?",
            workout_time_options,
            index=_safe_index(workout_time_options, str(defaults.get("workout_time", "30-45 minutes"))),
        )
        diet_preference = st.selectbox(
            "Diet preference",
            diet_preference_options,
            index=_safe_index(diet_preference_options, str(defaults.get("diet_preference", "No preference"))),
        )
        craving_level = st.selectbox(
            "Do you often crave sweets/snacks?",
            craving_options,
            index=_safe_index(craving_options, str(defaults.get("craving_level", "Sometimes"))),
        )
        stress_level = st.selectbox(
            "Stress level",
            stress_options,
            index=_safe_index(stress_options, str(defaults.get("stress_level", "Medium"))),
        )
        sleep_quality = st.selectbox(
            "Sleep quality",
            sleep_options,
            index=_safe_index(sleep_options, str(defaults.get("sleep_quality", "Average"))),
        )
        default_conditions = defaults.get("health_conditions", ["None"])
        if isinstance(default_conditions, str):
            default_conditions = [x.strip() for x in default_conditions.split(",") if x.strip()]
        default_conditions = [c for c in default_conditions if c in health_options]
        if not default_conditions:
            default_conditions = ["None"]
        st.markdown("**Do you have any injuries or health conditions?**")
        picked_conditions: list[str] = []
        pills_per_row = 4
        for row_start in range(0, len(health_options), pills_per_row):
            row_options = health_options[row_start : row_start + pills_per_row]
            row_cols = st.columns(len(row_options), gap="small")
            for idx, (col, option) in enumerate(zip(row_cols, row_options)):
                option_key = f"profile_health_condition_{row_start + idx}"
                with col:
                    is_selected = st.checkbox(
                        option,
                        value=option in default_conditions,
                        key=option_key,
                    )
                if is_selected:
                    picked_conditions.append(option)
        lifestyle_text = st.text_area(
            "Lifestyle description",
            value=defaults.get(
                "lifestyle_text",
                (
                    "I work night shifts, only have 20 minutes to work out on weekdays, "
                    "and prefer training at home. I also crave sweets late at night and have high stress."
                ),
            ),
            help="Include schedule, cravings, stress, sleep, equipment, and any injuries.",
        )
        submitted = st.form_submit_button("Save and Generate Plan")

    if not submitted:
        return None
    health_conditions = _normalize_health_conditions(picked_conditions)
    return {
        "age": age,
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "goal_weight_kg": goal_weight_kg,
        "goal_timeline_weeks": goal_timeline_weeks,
        "days_per_week": days_per_week,
        "lifestyle_text": lifestyle_text,
        "schedule_type": schedule_type,
        "workout_location": workout_location,
        "workout_time": workout_time,
        "diet_preference": diet_preference,
        "craving_level": craving_level,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "health_conditions": health_conditions,
    }


def render_workout_plan_tab(workouts: pd.DataFrame, lifestyle: dict) -> None:
    st.markdown('<div class="section-title">Recommended Weekly Exercise Plan</div>', unsafe_allow_html=True)
    total_weekly_time = float(workouts["duration_min"].sum()) if not workouts.empty else 0.0
    avg_duration = float(workouts["duration_min"].mean()) if not workouts.empty else 0.0
    top_equipment = (
        str(workouts["equipment"].mode().iat[0]).strip()
        if (not workouts.empty and "equipment" in workouts.columns and not workouts["equipment"].dropna().empty)
        else "bodyweight"
    )

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Weekly Time</div>
            <div class="metric-value">{total_weekly_time:.0f} min</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Session</div>
            <div class="metric-value">{avg_duration:.0f} min</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Equipment</div>
            <div class="metric-value">{top_equipment}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Daily Workout Cards</div>', unsafe_allow_html=True)
    if workouts.empty:
        st.info("No workout suggestions available yet.")
    else:
        for idx, row in workouts.reset_index(drop=True).iterrows():
            exercise_name = str(row.get("exercise_name", "Workout"))
            muscle_group = str(row.get("muscle_group", "full body"))
            equipment = str(row.get("equipment", "bodyweight"))
            duration_min = float(row.get("duration_min", 0))
            st.markdown(
                f"""
                <div class="block-card" style="border-left:4px solid #16a34a; margin-bottom:10px;">
                    <div class="small-note" style="margin-bottom:2px;">Day {idx + 1}</div>
                    <div style="font-size:1.85rem; font-weight:900; color:#14532d; margin-bottom:6px;">{exercise_name}</div>
                    <span class="chip">Muscle: {muscle_group}</span>
                    <span class="chip">Equipment: {equipment}</span>
                    <span class="chip">{duration_min:.0f} min</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        workout_table = workouts[["exercise_name", "muscle_group", "difficulty", "equipment", "duration_min"]].copy()
        workout_table.insert(0, "Day", [f"Day {i + 1}" for i in range(len(workout_table))])
        workout_table.columns = ["Day", "Exercise", "Muscle Group", "Difficulty", "Equipment", "Duration (min)"]
        with st.expander("View workout table", expanded=False):
            _render_mint_table(workout_table)

    if lifestyle["home_workout"]:
        st.info("This plan prioritizes home-friendly exercises based on your profile.")
    else:
        st.info("This plan balances gym/home exercise options based on your profile.")


def render_plan_screen(plan_data: dict) -> None:
    progress_pct = 100.0
    st.markdown(
        f"""
        <div class="result-shell">
            <div class="result-card">
                <div class="result-top-progress">
                    <div class="result-top-progress-fill" style="width:{progress_pct:.0f}%"></div>
                </div>
                <div class="result-ready-title">Your personalized result is ready.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:2.8rem'></div>", unsafe_allow_html=True)
    render_plan_header(
        predicted_body_type=plan_data["predicted_body_type"],
        bmi=plan_data["bmi"],
        calorie_target=plan_data["calorie_target"],
        similarity=plan_data["similarity"],
        weight_kg=plan_data["weight_kg"],
        goal_weight_kg=plan_data["goal_weight_kg"],
        days_per_week=plan_data["days_per_week"],
    )
    st.markdown("<div style='height:1.05rem'></div>", unsafe_allow_html=True)
    st.progress(1.0, text="Goal alignment progress: 100%")
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    render_twin_section(plan_data["twin"])
    tab_meals, tab_workouts, tab_lifestyle = st.tabs(["Diet Plan", "Lift Plan", "Lifestyle Fit"])
    with tab_meals:
        render_diet_plan_tab(plan_data["meals"])
    with tab_workouts:
        render_workout_plan_tab(plan_data["workouts"], plan_data["lifestyle"])
    with tab_lifestyle:
        render_lifestyle_fit_tab(
            plan_data["lifestyle"],
            plan_data.get("lifestyle_fit"),
            plan_data.get("lifestyle_recommendations"),
        )
    st.markdown(
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
