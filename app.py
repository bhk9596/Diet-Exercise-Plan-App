import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path(__file__).parent / "data"
AUTH_DB = DATA_DIR / "app_users.db"


@st.cache_data
def load_data():
    body_df = pd.read_csv(DATA_DIR / "nhanes_body_profiles.csv")
    diet_df = pd.read_csv(DATA_DIR / "diet_lifestyle_profiles.csv")
    gym_df = pd.read_csv(DATA_DIR / "megagym_subset.csv")
    food_df = pd.read_csv(DATA_DIR / "food_catalog.csv")
    activity_df = pd.read_csv(DATA_DIR / "activity_multipliers.csv")
    return body_df, diet_df, gym_df, food_df, activity_df


def apply_custom_theme():
    st.markdown(
        """
        <style>
            :root {
                --apple-black: #000000;
                --apple-bg-light: #f5f5f7;
                --apple-text: #1d1d1f;
                --apple-text-secondary: rgba(0, 0, 0, 0.8);
                --apple-text-tertiary: rgba(0, 0, 0, 0.48);
                --apple-blue: #0071e3;
                --apple-link: #0066cc;
                --apple-shadow: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
            }
            .stApp {
                background: var(--apple-bg-light);
                color: var(--apple-text);
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 16px;
            }
            header[data-testid="stHeader"] {
                display: none !important;
                height: 0 !important;
            }
            [data-testid="stToolbar"] {
                display: none !important;
            }
            .main .block-container {
                max-width: 980px;
                padding-top: 0.35rem;
                padding-bottom: 2rem;
            }
            [data-testid="stSidebar"] {
                background: var(--apple-bg-light);
                border-right: 1px solid rgba(0, 0, 0, 0.08);
                min-width: 220px !important;
                max-width: 220px !important;
            }
            h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
                color: var(--apple-text);
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
                letter-spacing: -0.374px;
            }
            .app-title-wrap {
                text-align: center;
                margin-bottom: 0.6rem;
            }
            .app-title {
                font-size: 2.5rem;
                font-weight: 600;
                letter-spacing: -0.28px;
                line-height: 1.1;
                margin-bottom: 0.25rem;
            }
            .app-subtitle {
                font-size: 1.06rem;
                line-height: 1.47;
                color: var(--apple-text-secondary);
                letter-spacing: -0.374px;
                text-align: center;
                margin-bottom: 0.6rem;
            }
            .hero-card {
                background: #ffffff;
                border: none;
                border-radius: 12px;
                box-shadow: rgba(0, 0, 0, 0.08) 0px 8px 24px;
                padding: 18px 20px;
                margin-bottom: 14px;
            }
            .hero-dark {
                background: #2c3138;
                color: #ffffff;
                border-radius: 12px;
                box-shadow: rgba(0, 0, 0, 0.14) 0px 10px 26px;
            }
            .hero-dark .hero-title,
            .hero-dark .hero-sub {
                color: #ffffff;
            }
            .hero-dark .hero-sub {
                opacity: 0.92;
            }
            .hero-title {
                font-size: 1.31rem;
                font-weight: 600;
                color: var(--apple-text);
                margin-bottom: 4px;
                line-height: 1.19;
                letter-spacing: 0.231px;
            }
            .hero-sub {
                color: var(--apple-text-secondary);
                font-size: 1rem;
                line-height: 1.47;
                margin-bottom: 10px;
            }
            .metric-card {
                background: #ffffff;
                border: none;
                border-radius: 12px;
                box-shadow: rgba(0, 0, 0, 0.08) 0px 8px 24px;
                padding: 14px 16px;
                margin-bottom: 8px;
            }
            .metric-label {
                font-size: 0.9rem;
                color: var(--apple-text-tertiary);
                letter-spacing: -0.224px;
                margin-bottom: 6px;
            }
            .metric-value {
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 600;
                color: var(--apple-text);
                letter-spacing: -0.28px;
            }
            .section-title {
                font-size: 1.31rem;
                font-weight: 600;
                color: var(--apple-text);
                margin: 10px 0;
                line-height: 1.19;
                letter-spacing: 0.231px;
            }
            .chip {
                display: inline-block;
                background: #fafafc;
                color: var(--apple-text-secondary);
                border: 1px solid rgba(0, 0, 0, 0.04);
                border-radius: 980px;
                padding: 2px 10px;
                margin: 3px 6px 0 0;
                font-size: 0.88rem;
                letter-spacing: -0.224px;
            }
            .hero-dark .chip {
                background: transparent;
                color: #8ec5ff;
                border: 1px solid #8ec5ff;
            }
            .small-note {
                color: var(--apple-text-secondary);
                font-size: 0.95rem;
                letter-spacing: -0.224px;
                line-height: 1.5;
            }
            .table-shell {
                background: #ffffff;
                border-radius: 12px;
                box-shadow: rgba(0, 0, 0, 0.06) 0px 4px 16px;
                padding: 8px 10px;
                margin-top: 6px;
                margin-bottom: 8px;
            }
            [data-testid="stTable"] table {
                font-size: 0.95rem !important;
                width: 100% !important;
                border-collapse: collapse !important;
            }
            [data-testid="stTable"] thead tr {
                background: #f2f2f7 !important;
            }
            [data-testid="stTable"] th,
            [data-testid="stTable"] td {
                padding: 10px 12px !important;
                border-bottom: 1px solid #e8e8ed !important;
                text-align: left !important;
            }
            [data-testid="stTable"] tbody tr:nth-child(even) {
                background: #fafafc !important;
            }
            [data-testid="stTable"] tbody tr:nth-child(odd) {
                background: #ffffff !important;
            }
            .block-card {
                background: var(--apple-bg-light);
                border: none;
                border-radius: 8px;
                padding: 14px 14px 10px 14px;
                margin-bottom: 12px;
            }
            .auth-card {
                background: #ffffff;
                border: none;
                border-radius: 8px;
                box-shadow: var(--apple-shadow);
                padding: 20px 20px 8px 20px;
                margin-top: 8px;
            }
            .auth-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 2px;
                color: #1d1d1f;
            }
            .auth-sub {
                color: var(--apple-text-secondary);
                font-size: 0.9rem;
                margin-bottom: 10px;
            }
            .auth-kicker {
                text-align: center;
                color: var(--apple-text-secondary);
                font-size: 0.98rem;
                font-weight: 600;
                margin: 0.3rem 0 0.6rem 0;
            }
            .auth-page-wrap {
                max-width: 720px;
                margin: 0 auto;
                padding-top: 0.25rem;
            }
            .auth-close {
                display: none !important;
            }
            .auth-headline {
                text-align: center;
                font-size: 3rem;
                font-weight: 800;
                letter-spacing: -0.28px;
                line-height: 1.1;
                text-transform: uppercase;
                margin-bottom: 1.8rem;
            }
            .auth-form-shell {
                margin: 0 auto;
            }
            .auth-form-shell label {
                text-transform: uppercase !important;
                font-size: 0.86rem !important;
                font-weight: 800 !important;
                letter-spacing: 0.2px !important;
                color: #151515 !important;
            }
            .auth-mode {
                max-width: 340px;
                margin: 0 auto 0.9rem auto;
            }
            .auth-form-shell div[data-baseweb="input"] > div {
                border: 1px solid #111111 !important;
                border-radius: 2px !important;
                box-shadow: none !important;
            }
            .auth-form-shell .stFormSubmitButton > button,
            .auth-form-shell .stFormSubmitButton > button[kind="primary"],
            .auth-form-shell [data-testid="stFormSubmitButton"] button {
                background: #7c3aed !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 4px !important;
                min-height: 48px !important;
                width: 100% !important;
                font-size: 1.05rem !important;
                font-weight: 700 !important;
            }
            .auth-form-shell .stFormSubmitButton > button:hover,
            .auth-form-shell [data-testid="stFormSubmitButton"] button:hover {
                background: #6d28d9 !important;
            }
            div[data-baseweb="input"] > div,
            div[data-baseweb="textarea"] > div,
            div[data-baseweb="select"] > div {
                background: #ffffff !important;
                border: 1px solid #d2d2d7 !important;
                border-radius: 11px !important;
                color: var(--apple-text) !important;
            }
            input, textarea {
                color: var(--apple-text) !important;
            }
            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                background: var(--apple-blue) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 400 !important;
                font-size: 17px !important;
                line-height: 1 !important;
                letter-spacing: 0 !important;
                padding: 8px 15px !important;
                transition: all 0.18s ease;
            }
            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button[kind="primary"]:hover {
                background: #1184f6 !important;
            }
            .stButton > button[kind="secondary"],
            .stFormSubmitButton > button[kind="secondary"] {
                background: #1d1d1f !important;
                color: #ffffff !important;
                border: 1px solid #d2d2d7 !important;
                border-radius: 8px !important;
                font-size: 17px !important;
                font-weight: 400 !important;
                line-height: 1 !important;
                padding: 8px 15px !important;
                transition: all 0.18s ease;
            }
            .stButton > button[kind="secondary"]:hover,
            .stFormSubmitButton > button[kind="secondary"]:hover {
                background: #000000 !important;
            }
            [data-baseweb="tab-list"] {
                gap: 8px;
            }
            [data-baseweb="tab"] {
                border-radius: 980px !important;
                background: transparent !important;
                color: var(--apple-link) !important;
                border: 1px solid var(--apple-link) !important;
                padding: 6px 14px !important;
                font-size: 14px !important;
                letter-spacing: -0.224px !important;
            }
            [aria-selected="true"][data-baseweb="tab"] {
                background: var(--apple-link) !important;
                color: #ffffff !important;
                font-weight: 400 !important;
            }
            .stProgress > div > div > div > div {
                background: var(--apple-blue) !important;
            }
            .stProgress > div > div > div {
                background-color: rgba(0, 0, 0, 0.12) !important;
            }
            .progress-shell {
                width: 100%;
                height: 12px;
                background: #ececf1;
                border: 1px solid #d8d8de;
                border-radius: 999px;
                overflow: hidden;
                margin-bottom: 0.7rem;
            }
            .progress-fill {
                height: 100%;
                background: var(--apple-blue);
                border-radius: 999px;
                transition: width 0.25s ease;
            }
            [data-baseweb="slider"] [role="slider"] {
                background: #0a84ff !important;
                border: 2px solid #ffffff !important;
                box-shadow: 0 0 0 3px #d9ecff;
                transition: transform 0.15s ease;
            }
            [data-baseweb="slider"] > div > div {
                background: #d1d1d6 !important;
            }
            [data-baseweb="slider"] [role="slider"]:hover {
                transform: scale(1.05);
            }
            .side-card {
                background: #ffffff;
                border: none;
                border-radius: 8px;
                box-shadow: var(--apple-shadow);
                padding: 10px 12px;
                margin-bottom: 10px;
            }
            .side-title {
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 3px;
            }
            .wizard-card {
                background: #ffffff;
                border: none;
                border-radius: 8px;
                box-shadow: var(--apple-shadow);
                padding: 1rem 1.1rem 0.9rem 1.1rem;
                margin-top: 0.35rem;
                transition: all 0.2s ease;
            }
            .wizard-step-label {
                font-size: 1.05rem;
                font-weight: 600;
                color: #3a3a3c;
                margin-top: 0.1rem;
            }
            .wizard-question {
                font-size: 1.7rem;
                font-weight: 700;
                color: #1d1d1f;
                margin: 0.3rem 0 0.3rem 0;
                line-height: 1.2;
            }
            .wizard-helper {
                color: var(--apple-text-secondary);
                font-size: 0.96rem;
                margin-bottom: 0.35rem;
            }
            .age-live-value {
                font-size: 1.2rem;
                font-weight: 700;
                color: #005ecb;
                margin-top: 0.15rem;
            }
            .top-right-actions + div .stButton > button {
                border-radius: 980px !important;
                background: transparent !important;
                color: var(--apple-link) !important;
                border: 1px solid var(--apple-link) !important;
                font-size: 14px !important;
                padding: 6px 14px !important;
                letter-spacing: -0.224px !important;
                line-height: 1.2 !important;
            }
            div[data-testid="stHorizontalBlock"] > div:has(.top-right-actions) {
                justify-content: flex-end;
            }
            .auth-form-shell div[data-testid="stFormSubmitButton"] button,
            .auth-form-shell .stFormSubmitButton > button,
            .auth-form-shell .stFormSubmitButton > button[kind="primary"] {
                background: #7c3aed !important;
                color: #ffffff !important;
                font-weight: 800 !important;
            }
            .auth-form-shell div[data-testid="stFormSubmitButton"] button:hover,
            .auth-form-shell .stFormSubmitButton > button:hover {
                background: #6d28d9 !important;
            }
            .auth-form-shell div[data-baseweb="input"] > div {
                background: #ffffff !important;
                border: 1px solid #111111 !important;
                border-radius: 4px !important;
                box-shadow: none !important;
            }
            .auth-form-shell div[data-baseweb="input"] > div > div {
                background: #ffffff !important;
            }
            .auth-form-shell .login-yellow [data-testid="stFormSubmitButton"] button,
            .auth-form-shell .login-yellow .stFormSubmitButton > button {
                background: #facc15 !important;
                color: #111111 !important;
                border: 1px solid #facc15 !important;
            }
            .auth-form-shell .login-yellow [data-testid="stFormSubmitButton"] button:hover,
            .auth-form-shell .login-yellow .stFormSubmitButton > button:hover {
                background: #eab308 !important;
                border-color: #eab308 !important;
            }
            /* Strong auth-only overrides */
            .auth-form-shell form button[kind],
            .auth-form-shell form div[data-testid="stFormSubmitButton"] > button,
            .auth-form-shell form .stFormSubmitButton > button {
                border-radius: 10px !important;
                min-height: 46px !important;
                font-size: 1.05rem !important;
                font-weight: 800 !important;
                box-shadow: none !important;
            }
            .auth-form-shell form#register_form div[data-testid="stFormSubmitButton"] > button,
            .auth-form-shell form#register_form .stFormSubmitButton > button {
                background: #7c3aed !important;
                color: #ffffff !important;
                border: 1px solid #7c3aed !important;
            }
            .auth-form-shell form#register_form div[data-testid="stFormSubmitButton"] > button:hover,
            .auth-form-shell form#register_form .stFormSubmitButton > button:hover {
                background: #6d28d9 !important;
                border-color: #6d28d9 !important;
            }
            .auth-form-shell .login-yellow form#login_form div[data-testid="stFormSubmitButton"] > button,
            .auth-form-shell .login-yellow form#login_form .stFormSubmitButton > button {
                background: #facc15 !important;
                color: #111111 !important;
                border: 1px solid #facc15 !important;
            }
            .auth-form-shell .login-yellow form#login_form div[data-testid="stFormSubmitButton"] > button:hover,
            .auth-form-shell .login-yellow form#login_form .stFormSubmitButton > button:hover {
                background: #eab308 !important;
                border-color: #eab308 !important;
            }
            .auth-form-shell div[data-baseweb="input"] {
                background: #ffffff !important;
                border-radius: 10px !important;
                overflow: hidden !important;
            }
            .auth-form-shell div[data-baseweb="input"] > div {
                border: 1px solid #111111 !important;
                border-radius: 10px !important;
                box-shadow: none !important;
                background: #ffffff !important;
            }
            .auth-form-shell div[data-baseweb="input"] button {
                background: #ffffff !important;
                color: #6b7280 !important;
                border: none !important;
                border-left: 1px solid #e5e7eb !important;
                border-radius: 0 !important;
            }
            .vesper-auth-wrap {
                max-width: 980px;
                margin: 0 auto;
                padding: 1.2rem 0 1.8rem 0;
            }
            .vesper-auth-hero {
                background: linear-gradient(135deg, #7c3aed 0%, #a855f7 45%, #6d28d9 100%);
                border-radius: 12px;
                min-height: 520px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: rgba(109, 40, 217, 0.25) 0px 14px 40px;
                position: relative;
                overflow: hidden;
            }
            .vesper-auth-hero::before {
                content: "";
                position: absolute;
                inset: 0;
                background: rgba(20, 12, 45, 0.24);
            }
            .vesper-auth-card {
                position: relative;
                z-index: 2;
                width: 360px;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: rgba(15, 23, 42, 0.22) 0px 16px 30px;
                padding: 12px 14px 14px 14px;
            }
            .vesper-auth-brand {
                text-align: center;
                color: #ffffff;
                position: absolute;
                top: 20px;
                left: 0;
                right: 0;
                z-index: 2;
                font-size: 1.08rem;
                font-weight: 700;
                letter-spacing: 0.6px;
            }
            .vesper-auth-tabs {
                margin-bottom: 8px;
            }
            .vesper-auth-tabs [role="radiogroup"] {
                gap: 0 !important;
                border-bottom: 1px solid #ebeaf0;
            }
            .vesper-auth-tabs label {
                text-transform: none !important;
                font-size: 0.82rem !important;
                font-weight: 700 !important;
                color: #4b5563 !important;
                padding-bottom: 6px !important;
            }
            .vesper-auth-card label {
                text-transform: none !important;
                font-size: 0.8rem !important;
                font-weight: 700 !important;
                color: #4b5563 !important;
                letter-spacing: 0 !important;
            }
            .vesper-auth-card div[data-baseweb="input"] > div {
                border: 1px solid #d8d8df !important;
                border-radius: 6px !important;
                min-height: 36px !important;
                box-shadow: none !important;
            }
            .vesper-auth-card [data-testid="stFormSubmitButton"] button {
                width: 100% !important;
                min-height: 34px !important;
                border-radius: 999px !important;
                border: none !important;
                background: linear-gradient(90deg, #6d28d9 0%, #a855f7 100%) !important;
                color: #ffffff !important;
                font-size: 0.83rem !important;
                font-weight: 700 !important;
                box-shadow: rgba(109, 40, 217, 0.32) 0px 8px 18px !important;
            }
            .vesper-auth-card [data-testid="stFormSubmitButton"] button:hover {
                background: linear-gradient(90deg, #5b21b6 0%, #9333ea 100%) !important;
            }
            .fit-auth-wrap {
                max-width: 1220px;
                margin: 0 auto;
                padding: 0.25rem 0 1rem 0;
            }
            .fit-auth-nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.65rem;
            }
            .fit-auth-brand {
                color: #0b1f6a;
                font-size: 2.0rem;
                font-weight: 800;
                letter-spacing: -0.3px;
                line-height: 1.05;
            }
            .fit-auth-menu {
                color: #0b1f6a;
                font-size: 0.98rem;
                font-weight: 600;
                display: flex;
                gap: 28px;
                align-items: center;
            }
            .fit-auth-menu-chip {
                background: #0b1f6a;
                color: #ffffff;
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 700;
            }
            .fit-auth-right {
                padding: 0.95rem 0 0 1.6rem;
            }
            .fit-auth-kicker {
                color: #0b1f6a;
                font-size: 1.01rem;
                font-weight: 600;
                margin-bottom: 0.6rem;
            }
            .fit-auth-title {
                color: #0b1f6a;
                font-size: 4.0rem;
                font-weight: 800;
                letter-spacing: -1.2px;
                line-height: 1.05;
                margin-bottom: 0.8rem;
            }
            .fit-auth-desc {
                color: #0b1f6a;
                font-size: 1.05rem;
                line-height: 1.55;
                max-width: 92%;
                margin-bottom: 0.9rem;
            }
            .fit-auth-cta-row .stButton > button {
                background: #0b1f6a !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                min-height: 46px !important;
                font-weight: 800 !important;
                width: 100% !important;
            }
            .fit-auth-cta-row .stButton > button:hover {
                background: #14339a !important;
            }
            div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
                gap: 10px !important;
            }
            div[data-testid="stHorizontalBlock"] .st-key-auth_login_btn button,
            div[data-testid="stHorizontalBlock"] .st-key-auth_register_btn button {
                background: #0b1f6a !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                min-height: 44px !important;
                font-weight: 800 !important;
                width: 100% !important;
            }
            div[data-testid="stHorizontalBlock"] .st-key-auth_login_btn button:hover,
            div[data-testid="stHorizontalBlock"] .st-key-auth_register_btn button:hover {
                background: #14339a !important;
            }
            .fit-auth-form-shell {
                max-width: 420px;
                margin-top: 0.65rem;
            }
            .fit-auth-form-shell label {
                color: #0f172a !important;
                font-size: 0.92rem !important;
                font-weight: 700 !important;
                text-transform: none !important;
                letter-spacing: 0 !important;
            }
            .fit-auth-form-shell div[data-baseweb="input"] > div {
                border: 1px solid #d3d8e3 !important;
                border-radius: 10px !important;
                min-height: 42px !important;
                box-shadow: none !important;
            }
            .fit-auth-form-shell [data-testid="stFormSubmitButton"] button {
                background: #0b1f6a !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                min-height: 44px !important;
                width: 100% !important;
                font-weight: 800 !important;
            }
            .fit-auth-form-shell [data-testid="stFormSubmitButton"] button:hover {
                background: #14339a !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_auth_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                age INTEGER NOT NULL,
                sex TEXT NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                goal_weight_kg REAL NOT NULL,
                days_per_week INTEGER NOT NULL,
                lifestyle_text TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return actual == expected


def create_user(username: str, password: str):
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with sqlite3.connect(AUTH_DB) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.strip(), hash_password(password)),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "That username already exists."
    return True, "Account created successfully."


def authenticate_user(username: str, password: str):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row:
        return None
    user_id, password_hash = row
    return user_id if verify_password(password, password_hash) else None


def save_user_profile(user_id: int, profile: dict):
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, age, sex, height_cm, weight_kg, goal_weight_kg, days_per_week, lifestyle_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
                sex = excluded.sex,
                height_cm = excluded.height_cm,
                weight_kg = excluded.weight_kg,
                goal_weight_kg = excluded.goal_weight_kg,
                days_per_week = excluded.days_per_week,
                lifestyle_text = excluded.lifestyle_text
            """,
            (
                user_id,
                profile["age"],
                profile["sex"],
                profile["height_cm"],
                profile["weight_kg"],
                profile["goal_weight_kg"],
                profile["days_per_week"],
                profile["lifestyle_text"],
            ),
        )
        conn.commit()


def load_user_profile(user_id: int):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            """
            SELECT age, sex, height_cm, weight_kg, goal_weight_kg, days_per_week, lifestyle_text
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    keys = ["age", "sex", "height_cm", "weight_kg", "goal_weight_kg", "days_per_week", "lifestyle_text"]
    return dict(zip(keys, row))


def parse_lifestyle(text: str):
    t = text.lower()
    tags = {
        "night_shift": int(any(k in t for k in ["night shift", "overnight", "late shift"])),
        "sugar_craving": int(any(k in t for k in ["sugar", "dessert", "sweet", "crave", "snack at night"])),
        "home_workout": int(any(k in t for k in ["home", "apartment", "no gym"])),
        "vegetarian_pref": int(any(k in t for k in ["vegetarian", "plant-based", "vegan"])),
        "high_stress": int(any(k in t for k in ["stress", "busy", "anxious", "burnout", "overwhelmed"])),
        "short_sessions": int(any(k in t for k in ["20 minutes", "15 minutes", "short workout", "quick workout"])),
        "low_sleep": int(any(k in t for k in ["sleep 5", "sleep 4", "insomnia", "poor sleep"])),
        "injury_care": int(any(k in t for k in ["injury", "knee pain", "back pain", "joint pain"])),
    }
    matched_cues = []
    if tags["night_shift"]:
        matched_cues.append("you mentioned shift/late-hour routines")
    if tags["short_sessions"]:
        matched_cues.append("you prefer short workout windows")
    if tags["home_workout"]:
        matched_cues.append("you prefer home-based training")
    if tags["sugar_craving"]:
        matched_cues.append("you flagged sugar/snack cravings")
    if tags["high_stress"]:
        matched_cues.append("you described high stress or burnout")
    if tags["low_sleep"]:
        matched_cues.append("you reported limited or poor sleep")
    if tags["injury_care"]:
        matched_cues.append("you mentioned pain/injury constraints")
    return {
        **tags,
        "matched_cues": matched_cues,
    }


def get_activity_multiplier(days_per_week: int, activity_df: pd.DataFrame):
    row = activity_df.loc[activity_df["days_per_week"] == days_per_week]
    if row.empty:
        return 1.2
    return float(row.iloc[0]["multiplier"])


def estimate_calories(age, sex, height_cm, weight_kg, goal_weight_kg, days_per_week, activity_df):
    sex_offset = 5 if sex == "M" else -161
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + sex_offset
    tdee = bmr * get_activity_multiplier(days_per_week, activity_df)
    weight_gap = goal_weight_kg - weight_kg
    if weight_gap > 0:
        target = tdee + min(450, 120 * weight_gap)
    elif weight_gap < 0:
        target = tdee - min(500, 130 * abs(weight_gap))
    else:
        target = tdee
    return max(1200, round(target))


def build_body_classifier(body_df: pd.DataFrame):
    x = body_df[["age", "height_cm", "weight_kg", "sex_bin"]]
    y = body_df["body_type"]
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(x, y)
    return clf


def retrieve_diet_twin(user_vector: np.ndarray, diet_df: pd.DataFrame):
    feature_cols = [
        "age",
        "height_cm",
        "weight_kg",
        "sex_bin",
        "night_shift",
        "sugar_craving",
        "home_workout",
        "vegetarian_pref",
        "high_stress",
        "short_sessions",
        "goal_direction",
    ]
    # Higher weights on lifestyle features to emphasize behavior-fit over pure body stats.
    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.3, 2.6, 2.2, 1.8, 2.2, 2.2, 1.5], dtype=float)
    weighted_matrix = diet_df[feature_cols].values * weights
    weighted_user = user_vector * weights
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(weighted_matrix)
    distances, indices = nn.kneighbors(weighted_user.reshape(1, -1))
    best_idx = indices[0][0]
    similarity = 1 - distances[0][0]
    return diet_df.iloc[best_idx], float(similarity)


def pick_meals(food_df: pd.DataFrame, calorie_target: int, vegetarian_pref: int):
    candidates = food_df.copy()
    if vegetarian_pref:
        candidates = candidates[candidates["vegetarian"] == 1]
    candidates = candidates.sort_values(by="protein_g", ascending=False)
    daily_budget = calorie_target
    meal_plan = []
    for meal_type, share in [("breakfast", 0.28), ("lunch", 0.34), ("dinner", 0.30), ("snack", 0.08)]:
        target = daily_budget * share
        row = (candidates.iloc[(candidates["calories"] - target).abs().argsort()]).head(1)
        meal_plan.append(row.assign(meal_type=meal_type))
    return pd.concat(meal_plan, ignore_index=True)


def pick_workouts(gym_df: pd.DataFrame, home_workout: int, short_sessions: int, days_per_week: int):
    d = gym_df.copy()
    if home_workout:
        d = d[d["equipment"].isin(["bodyweight", "dumbbell", "resistance_band", "bands"])]
    if short_sessions:
        d = d[d["duration_min"] <= 25]
    if d.empty:
        d = gym_df.copy()
    weekly_count = min(max(days_per_week, 3), 6)
    return d.sort_values(by=["difficulty", "duration_min"]).head(weekly_count)


def render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df):
    classifier = build_body_classifier(body_df)

    age = int(profile["age"])
    sex = profile["sex"]
    height_cm = float(profile["height_cm"])
    weight_kg = float(profile["weight_kg"])
    goal_weight_kg = float(profile["goal_weight_kg"])
    days_per_week = int(profile["days_per_week"])
    lifestyle_text = profile["lifestyle_text"]

    lifestyle = parse_lifestyle(lifestyle_text)
    sex_bin = 1 if sex == "M" else 0
    predicted_body_type = classifier.predict([[age, height_cm, weight_kg, sex_bin]])[0]
    calorie_target = estimate_calories(
        age, sex, height_cm, weight_kg, goal_weight_kg, days_per_week, activity_df
    )
    goal_direction = int(goal_weight_kg > weight_kg) - int(goal_weight_kg < weight_kg)

    user_vector = np.array(
        [
            age,
            height_cm,
            weight_kg,
            sex_bin,
            lifestyle["night_shift"],
            lifestyle["sugar_craving"],
            lifestyle["home_workout"],
            lifestyle["vegetarian_pref"],
            lifestyle["high_stress"],
            lifestyle["short_sessions"],
            goal_direction,
        ],
        dtype=float,
    )
    twin, similarity = retrieve_diet_twin(user_vector, diet_df)
    meals = pick_meals(food_df, calorie_target, lifestyle["vegetarian_pref"])
    workouts = pick_workouts(gym_df, lifestyle["home_workout"], lifestyle["short_sessions"], days_per_week)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    goal_progress = max(0.0, min(1.0, 1.0 - abs(goal_weight_kg - weight_kg) / max(weight_kg, 1)))

    st.markdown(
        f"""
        <div class="hero-card hero-dark">
            <div class="hero-title">Personalized Plan</div>
            <div class="hero-sub">Lifestyle-weighted guidance based on your profile, goals, and routine constraints.</div>
            <span class="chip">Body Type: {predicted_body_type}</span>
            <span class="chip">BMI: {bmi:.1f}</span>
            <span class="chip">Daily Calories: {calorie_target} kcal</span>
            <span class="chip">Diet Twin Match: {similarity:.1%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(goal_progress, text=f"Goal alignment progress: {goal_progress * 100:.0f}%")

    c1, c2, c3 = st.columns(3)
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

    st.markdown('<div class="section-title">Your Closest Diet Twin</div>', unsafe_allow_html=True)
    st.write(
        f"Matched profile: **{twin['profile_id']}** | Pattern: **{twin['diet_pattern']}** | "
        f"Avg adherence: **{twin['adherence_score']:.0f}%**"
    )
    st.markdown(
        f"""
        <div class="block-card">
            <div>{twin["notes"]}</div>
            <div class="small-note" style="margin-top:6px;">
                Lifestyle-weighted matching is prioritized so recommendations reflect your daily reality, not just body stats.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_meals, tab_workouts, tab_lifestyle = st.tabs(["Diet Plan", "Lift Plan", "Lifestyle Fit"])

    with tab_meals:
        st.markdown('<div class="section-title">Recommended Meal Structure</div>', unsafe_allow_html=True)
        meal_table = meals[["meal_type", "food_name", "calories", "protein_g", "carbs_g", "fat_g"]].copy()
        meal_table.columns = ["Meal", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
        st.markdown('<div class="table-shell">', unsafe_allow_html=True)
        st.table(meal_table)
        st.markdown('</div>', unsafe_allow_html=True)
        total_protein = meals["protein_g"].sum()
        st.markdown(
            f"""
            <div class="small-note">
                Meals are selected by calorie fit and protein priority.
                Estimated daily protein from this meal set: <b>{total_protein:.0f} g</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_workouts:
        st.markdown('<div class="section-title">Recommended Weekly Lift Plan</div>', unsafe_allow_html=True)
        workout_table = workouts[["exercise_name", "muscle_group", "difficulty", "equipment", "duration_min"]].copy()
        workout_table.columns = ["Exercise", "Muscle Group", "Difficulty", "Equipment", "Duration (min)"]
        st.markdown('<div class="table-shell">', unsafe_allow_html=True)
        st.table(workout_table)
        st.markdown('</div>', unsafe_allow_html=True)
        avg_duration = workouts["duration_min"].mean() if not workouts.empty else 0
        st.markdown(
            f"""
            <div class="small-note">
                Workouts honor schedule and equipment constraints.
                Average session duration: <b>{avg_duration:.0f} minutes</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_lifestyle:
        st.markdown('<div class="section-title">Why This Plan Fits Your Lifestyle</div>', unsafe_allow_html=True)
        cues = lifestyle["matched_cues"]
        if cues:
            st.markdown("- " + "\n- ".join([f"We adjusted recommendations because {c}." for c in cues[:5]]))
        else:
            st.markdown(
                "- We used your body profile and goal to personalize the baseline plan.\n"
                "- Add more daily routine details in the lifestyle box for deeper personalization."
            )
        if lifestyle["injury_care"] or lifestyle["low_sleep"]:
            st.info("Recovery-sensitive mode is on: prioritize consistency and moderate intensity.")


def render_onboarding_form(existing_profile=None):
    defaults = existing_profile or {}
    with st.form("profile_form"):
        st.subheader("Tell us about you")
        age = st.slider("Age", 16, 75, int(defaults.get("age", 24)))
        sex = st.selectbox("Sex", ["M", "F"], index=0 if defaults.get("sex", "M") == "M" else 1)
        height_cm = st.slider("Height (cm)", 145, 210, int(float(defaults.get("height_cm", 172))))
        weight_kg = st.slider("Current weight (kg)", 40, 160, int(float(defaults.get("weight_kg", 75))))
        goal_weight_kg = st.slider("Goal weight (kg)", 40, 160, int(float(defaults.get("goal_weight_kg", 68))))
        days_per_week = st.slider(
            "Workout days / week", 0, 7, int(defaults.get("days_per_week", 4))
        )
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
    return {
        "age": age,
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "goal_weight_kg": goal_weight_kg,
        "days_per_week": days_per_week,
        "lifestyle_text": lifestyle_text,
    }


def render_onboarding_wizard():
    steps = [
        ("age", "How old are you?"),
        ("sex", "What is your sex?"),
        ("height_cm", "What is your height (cm)?"),
        ("weight_kg", "What is your current weight (kg)?"),
        ("goal_weight_kg", "What is your goal weight (kg)?"),
        ("days_per_week", "How many days per week can you work out?"),
        ("lifestyle_text", "Describe your daily lifestyle"),
    ]
    defaults = {
        "age": 24,
        "sex": "M",
        "height_cm": 172,
        "weight_kg": 75,
        "goal_weight_kg": 68,
        "days_per_week": 4,
        "lifestyle_text": (
            "I work night shifts, only have 20 minutes to work out on weekdays, "
            "and prefer training at home. I also crave sweets late at night and have high stress."
        ),
    }

    if "onboarding_step" not in st.session_state:
        st.session_state["onboarding_step"] = 0
    if "onboarding_profile" not in st.session_state:
        st.session_state["onboarding_profile"] = defaults.copy()

    current_step = st.session_state["onboarding_step"]
    profile = st.session_state["onboarding_profile"]
    field, prompt = steps[current_step]
    labels = {
        "age": "A",
        "sex": "S",
        "height_cm": "H",
        "weight_kg": "W",
        "goal_weight_kg": "G",
        "days_per_week": "D",
        "lifestyle_text": "L",
    }

    st.markdown('<div class="wizard-card">', unsafe_allow_html=True)
    progress_pct = ((current_step + 1) / len(steps)) * 100
    st.markdown(
        f"""
        <div class="progress-shell">
            <div class="progress-fill" style="width:{progress_pct:.1f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="wizard-step-label">{labels.get(field, "S")} Step {current_step + 1} of {len(steps)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="wizard-question">{prompt}</div>', unsafe_allow_html=True)
    st.markdown('<div class="wizard-helper">Tell us a bit about yourself so we can personalize your plan.</div>', unsafe_allow_html=True)

    with st.form(f"onboarding_step_{current_step}", border=False):
        if field == "age":
            value = st.slider("Age", 16, 75, int(profile["age"]))
            st.markdown(f'<div class="age-live-value">Age: {value}</div>', unsafe_allow_html=True)
        elif field == "sex":
            value = st.selectbox("Sex", ["M", "F"], index=0 if profile["sex"] == "M" else 1)
        elif field == "height_cm":
            value = st.slider("Height (cm)", 145, 210, int(float(profile["height_cm"])))
        elif field == "weight_kg":
            value = st.slider("Current weight (kg)", 40, 160, int(float(profile["weight_kg"])))
        elif field == "goal_weight_kg":
            value = st.slider("Goal weight (kg)", 40, 160, int(float(profile["goal_weight_kg"])))
        elif field == "days_per_week":
            value = st.slider("Workout days / week", 0, 7, int(profile["days_per_week"]))
        else:
            value = st.text_area(
                "Lifestyle description",
                value=profile["lifestyle_text"],
                help="Include schedule, cravings, stress, sleep, equipment, and any injuries.",
            )

        c1, c2 = st.columns([1, 1])
        back_clicked = c1.form_submit_button("Back", disabled=current_step == 0, type="secondary")
        next_label = "Save and Generate Plan" if current_step == len(steps) - 1 else "Next"
        next_clicked = c2.form_submit_button(next_label, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if back_clicked and current_step > 0:
        st.session_state["onboarding_step"] = current_step - 1
        st.rerun()

    if next_clicked:
        profile[field] = value
        st.session_state["onboarding_profile"] = profile
        if current_step < len(steps) - 1:
            st.session_state["onboarding_step"] = current_step + 1
            st.rerun()

        final_profile = st.session_state["onboarding_profile"].copy()
        bmi = final_profile["weight_kg"] / ((final_profile["height_cm"] / 100) ** 2)
        st.info(f"Captured BMI: **{bmi:.1f}**")
        st.session_state.pop("onboarding_step", None)
        st.session_state.pop("onboarding_profile", None)
        return final_profile

    return None


def render_auth():
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "Login"
    st.markdown(
        """
        <div class="fit-auth-wrap">
            <div class="fit-auth-nav">
                <div class="fit-auth-brand">Diet Twin Planner</div>
                <div class="fit-auth-menu">
                    <span>About</span>
                    <span>Programs</span>
                    <span>Results</span>
                    <span class="fit-auth-menu-chip">Book a Plan</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.02, 1], gap="large")
    with left:
        st.image(
            "/Users/bztr1ng2l1ve/.cursor/projects/Users-bztr1ng2l1ve-Desktop-ml-diet-twin-app/assets/Screenshot_2026-04-19_at_15.59.32-36fadb4b-686f-4e52-a6f6-fe7a3a4b2b07.png",
            use_container_width=True,
        )
    with right:
        st.markdown('<div class="fit-auth-right">', unsafe_allow_html=True)
        st.markdown('<div class="fit-auth-kicker">HI, WELCOME TO DIET TWIN COACHING</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fit-auth-title">Your online fitness coach and nutrition planner.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="fit-auth-desc">Build better habits with practical workouts, smarter meal guidance, and a personalized plan designed for your real schedule.</div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Login", key="auth_login_btn"):
                st.session_state["auth_mode"] = "Login"
        with b2:
            if st.button("Register", key="auth_register_btn"):
                st.session_state["auth_mode"] = "Register"
        st.markdown('<div class="fit-auth-form-shell">', unsafe_allow_html=True)

        do_register = False
        do_login = False
        mode = st.session_state.get("auth_mode", "Login")
        if mode == "Register":
            with st.form("register_form"):
                username = st.text_input("Name", key="register_username", placeholder="Name")
                email_alias = st.text_input("Email", key="register_email", placeholder="Email")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="register_password",
                    placeholder="Password",
                )
                do_register = st.form_submit_button("Create account", type="primary", use_container_width=True)
        else:
            with st.form("login_form"):
                username = st.text_input("Email", key="login_username", placeholder="Email")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Password",
                )
                do_login = st.form_submit_button("Log in", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if do_login:
        user_id = authenticate_user(username, password)
        if user_id:
            st.session_state["user_id"] = user_id
            st.session_state["username"] = username.strip()
            st.success("Logged in.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    if do_register:
        if not email_alias.strip():
            st.error("Please enter your email.")
        else:
            ok, message = create_user(username, password)
            if ok:
                user_id = authenticate_user(username, password)
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username.strip()
                st.success("Account created. Please complete your profile.")
                st.rerun()
            else:
                st.error(message)


def render_top_right_logout():
    _, right = st.columns([0.86, 0.14])
    with right:
        st.markdown('<div class="top-right-actions"></div>', unsafe_allow_html=True)
        if st.button("Log out", key="logout_top_right", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def main():
    st.set_page_config(page_title="Diet Twin Planner", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_theme()
    init_auth_db()

    if "user_id" not in st.session_state:
        render_auth()
        return

    st.markdown(
        """
        <div class="app-title-wrap">
            <div class="app-title">Diet Twin Planner</div>
            <div class="app-subtitle">Premium, minimal planning interface with focused nutrition and training decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_top_right_logout()

    body_df, diet_df, gym_df, food_df, activity_df = load_data()
    user_id = st.session_state["user_id"]
    username = st.session_state.get("username", "user")

    profile = load_user_profile(user_id)
    if profile is None:
        new_profile = render_onboarding_wizard()
        if new_profile:
            save_user_profile(user_id, new_profile)
            st.success("Profile saved. Building your first plan...")
            st.rerun()
        return

    st.subheader("Your Saved Profile")
    with st.expander("Edit profile"):
        updated = render_onboarding_form(existing_profile=profile)
        if updated:
            save_user_profile(user_id, updated)
            profile = updated
            st.success("Profile updated.")

    render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df)

if __name__ == "__main__":
    main()
