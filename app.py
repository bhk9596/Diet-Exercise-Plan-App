import base64
import html
import io
import shutil
import textwrap
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

from diet_twin_finder import DietTwinFinder
from meal_generator import MealGenerator


DATA_DIR = Path(__file__).parent / "data"
_ONBOARDING_GENDER_AVATAR_WIDTH_PX = 288
WELCOME_VIDEO = Path(r"c:\Users\Administrator\Downloads\4777126_Woman_Runner_3840x2160.mp4")
WELCOME_FALLBACK_IMAGE = Path(
    "/Users/bztr1ng2l1ve/.cursor/projects/Users-bztr1ng2l1ve-Desktop-ml-diet-twin-app/assets/image-f734546c-2013-4351-aaf2-4302d0c6611d.png"
)


@st.cache_data
def load_data():
    body_df = pd.read_csv(DATA_DIR / "nhanes_body_profiles.csv")
    diet_df = pd.read_csv(DATA_DIR / "diet_lifestyle_profiles.csv")
    gym_df = pd.read_csv(DATA_DIR / "megagym_subset.csv")
    food_df = pd.read_csv(DATA_DIR / "clean_food_catalog.csv")
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
                color: #1d1d1f !important;
            }
            /* Tabs + tables: Streamlit theme can set light text on light bg (invisible rows). */
            div[data-testid="stTabs"] [role="tabpanel"] {
                color: #1d1d1f !important;
            }
            [data-testid="stTable"] table {
                font-size: 0.95rem !important;
                width: 100% !important;
                border-collapse: collapse !important;
                color: #1d1d1f !important;
            }
            [data-testid="stTable"] thead tr {
                background: #f2f2f7 !important;
            }
            [data-testid="stTable"] th,
            [data-testid="stTable"] td {
                padding: 10px 12px !important;
                border-bottom: 1px solid #e8e8ed !important;
                text-align: left !important;
                color: #1d1d1f !important;
            }
            [data-testid="stTable"] tbody tr:nth-child(even) {
                background: #fafafc !important;
            }
            [data-testid="stTable"] tbody tr:nth-child(odd) {
                background: #ffffff !important;
            }
            [data-testid="stTable"] tbody tr:nth-child(even) td,
            [data-testid="stTable"] tbody tr:nth-child(odd) td {
                color: #1d1d1f !important;
            }
            [data-testid="stDataFrame"] {
                color: #1d1d1f !important;
            }
            [data-testid="stDataFrame"] * {
                --gdg-text-color: #1d1d1f !important;
            }
            .block-card {
                background: var(--apple-bg-light);
                border: none;
                border-radius: 8px;
                padding: 14px 14px 10px 14px;
                margin-bottom: 12px;
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
            .welcome-bleed {
                position: fixed;
                inset: 0;
                width: 100vw;
                height: 100svh;
                margin: 0 !important;
                z-index: 1;
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
            .welcome-nav {
                display: flex;
                gap: 34px;
                font-size: 1.04rem;
                font-weight: 600;
            }
            .welcome-nav span {
                color: #ffffff;
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
                goal_timeline_weeks INTEGER NOT NULL,
                days_per_week INTEGER NOT NULL,
                schedule_type TEXT NOT NULL,
                workout_location TEXT NOT NULL,
                workout_time TEXT NOT NULL,
                diet_preference TEXT NOT NULL,
                craving_level TEXT NOT NULL,
                stress_level TEXT NOT NULL,
                sleep_quality TEXT NOT NULL,
                health_conditions TEXT NOT NULL,
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
            INSERT INTO user_profiles (
                user_id, age, sex, height_cm, weight_kg, goal_weight_kg, goal_timeline_weeks,
                days_per_week, schedule_type, workout_location, workout_time, 
                diet_preference, craving_level, stress_level, sleep_quality, health_conditions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
                sex = excluded.sex,
                height_cm = excluded.height_cm,
                weight_kg = excluded.weight_kg,
                goal_weight_kg = excluded.goal_weight_kg,
                goal_timeline_weeks = excluded.goal_timeline_weeks,
                days_per_week = excluded.days_per_week,
                schedule_type = excluded.schedule_type,
                workout_location = excluded.workout_location,
                workout_time = excluded.workout_time,
                diet_preference = excluded.diet_preference,
                craving_level = excluded.craving_level,
                stress_level = excluded.stress_level,
                sleep_quality = excluded.sleep_quality,
                health_conditions = excluded.health_conditions
            """,
            (
                user_id,
                profile["age"],
                profile["sex"],
                profile["height_cm"],
                profile["weight_kg"],
                profile["goal_weight_kg"],
                profile.get("goal_timeline_weeks", 12),
                profile["days_per_week"],
                profile["schedule_type"],
                profile["workout_location"],
                profile["workout_time"],
                profile["diet_preference"],
                profile["craving_level"],
                profile["stress_level"],
                profile["sleep_quality"],
                ",".join(profile.get("health_conditions", ["None"])),
            ),
        )
        conn.commit()


def load_user_profile(user_id: int):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            """
            SELECT age, sex, height_cm, weight_kg, goal_weight_kg, goal_timeline_weeks,
            days_per_week, schedule_type, workout_location, workout_time, 
            diet_preference, craving_level, stress_level, sleep_quality, health_conditions
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return None
    keys = ["age", "sex", "height_cm", "weight_kg", "goal_weight_kg", "goal_timeline_weeks",
            "days_per_week", "schedule_type", "workout_location", "workout_time",
            "diet_preference", "craving_level", "stress_level", "sleep_quality", "health_conditions"
    ]
    profile = dict(zip(keys, row))
    profile["health_conditions"] = profile["health_conditions"].split(",")
    return profile

def parse_lifestyle(profile: dict):
    schedule_type = profile.get("schedule_type", "Regular daytime")
    workout_location = profile.get("workout_location", "Home")
    workout_time = profile.get("workout_time", "30-45 minutes")
    diet_preference = profile.get("diet_preference", "No preference")
    craving_level = profile.get("craving_level", "Sometimes")
    stress_level = profile.get("stress_level", "Medium")
    sleep_quality = profile.get("sleep_quality", "Average")
    health_conditions = profile.get("health_conditions", ["None"])
    
    tags = {
        "night_shift": int(schedule_type == "Night shift"),
        "sugar_craving": int(craving_level == "Often"),
        "home_workout": int(workout_location == "Home"),
        "vegetarian_pref": int(diet_preference == "Vegetarian"),
        "high_stress": int(stress_level == "High"),
        "short_sessions": int(workout_time == "15-20 minutes"),
        "low_sleep": int(sleep_quality == "Poor"),
        "injury_care": int(any(c in health_conditions for c in [
            "Knee pain", "Back pain", "Shoulder injury", "Joint pain", "Severe Arthritis"
        ])),
        "medical_condition": int("None" not in health_conditions),
    }
    matched_cues = []
    if tags["night_shift"]:
        matched_cues.append("you have a night shift schedule")
    if tags["short_sessions"]:
        matched_cues.append("you prefer short workout sessions")
    if tags["home_workout"]:
        matched_cues.append("you prefer home-based training")
    if tags["sugar_craving"]:
        matched_cues.append("you often crave sweets or snacks")
    if tags["high_stress"]:
        matched_cues.append("you have high stress")
    if tags["low_sleep"]:
        matched_cues.append("you reported poor sleep")
    if tags["injury_care"]:
        matched_cues.append("you have an injury or pain constraint")
    if tags["medical_condition"]:
        matched_cues.append("you have a medical condition")
    return {
        **tags,
        "matched_cues": matched_cues,
    }
    
def process_user_inputs(profile: dict):
    age = int(profile["age"])
    sex = profile["sex"]
    height_cm = float(profile["height_cm"])
    weight_kg = float(profile["weight_kg"])
    goal_weight_kg = float(profile["goal_weight_kg"])
    goal_timeline_weeks = int(profile.get("goal_timeline_weeks", 12))
    days_per_week = int(profile["days_per_week"])

    sex_bin = 1 if sex == "M" else 0
    bmi = weight_kg / ((height_cm / 100) ** 2)
    goal_direction = int(goal_weight_kg > weight_kg) - int(goal_weight_kg < weight_kg)
    lifestyle = parse_lifestyle(profile)

    return {
        "age": age,
        "sex": sex,
        "sex_bin": sex_bin,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "goal_weight_kg": goal_weight_kg,
        "goal_timeline_weeks": goal_timeline_weeks,
        "days_per_week": days_per_week,
        "bmi": bmi,
        "goal_direction": goal_direction,
        **lifestyle,
    }

def get_activity_multiplier(days_per_week: int, activity_df: pd.DataFrame):
    row = activity_df.loc[activity_df["days_per_week"] == days_per_week]
    if row.empty:
        return 1.2
    return float(row.iloc[0]["multiplier"])


def estimate_calories(age, sex, height_cm, weight_kg, goal_weight_kg, goal_timeline_weeks, days_per_week, activity_df):
    sex_offset = 5 if sex == "M" else -161
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + sex_offset
    tdee = bmr * get_activity_multiplier(days_per_week, activity_df)
    weight_gap = goal_weight_kg - weight_kg
    weekly_change = weight_gap / goal_timeline_weeks
    calorie_adjustment = weekly_change * 7700 / 7
    if weight_gap > 0:
        calorie_adjustment = min(450, calorie_adjustment)
    elif weight_gap < 0:
        calorie_adjustment = max(-500, calorie_adjustment)
    else:
        calorie_adjustment = 0
    target = tdee + calorie_adjustment
    return max(1200, round(target))


def build_body_classifier(body_df: pd.DataFrame):
    x = body_df[["age", "height_cm", "weight_kg", "sex_bin"]]
    y = body_df["body_type"]
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(x, y)
    return clf


# --- Diet Twin and Meal Generation ---
# These capabilities are provided by the DietTwinFinder and MealGenerator classes.
# See diet_twin_finder.py and meal_generator.py for the algorithm implementations.

# Shared constants for the k-NN feature space
DIET_FEATURE_COLS = [
    "age", "height_cm", "weight_kg", "sex_bin",
    "night_shift", "sugar_craving", "home_workout",
    "vegetarian_pref", "high_stress", "short_sessions",
    "goal_direction",
]
# Weights applied AFTER StandardScaler — lifestyle flags dominate over body stats
DIET_FEATURE_WEIGHTS = np.array(
    [1.0, 1.0, 1.0, 1.0, 2.3, 2.6, 2.2, 1.8, 2.2, 2.2, 1.5], dtype=float
)


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

    st.html(
        textwrap.dedent(
            """
        <style>
            /* Full-bleed welcome mode: remove Streamlit container gutters */
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
            /* Hide any leftover Streamlit buttons on welcome screen */
            .stButton,
            [data-testid="stFormSubmitButton"] {
                display: none !important;
            }
        </style>
        """
        )
    )

    video_uri = get_welcome_video_data_uri()
    image_uri = get_welcome_bg_data_uri()
    if image_uri:
        fallback_style = f"background-image: url('{image_uri}'); background-size: cover; background-position: center center;"
    else:
        fallback_style = "background: linear-gradient(120deg, #1f2937, #0f172a);"
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
    return False


def render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df):
    classifier = build_body_classifier(body_df)

    processed = process_user_inputs(profile)

    age = processed["age"]
    sex = processed["sex"]
    height_cm = processed["height_cm"]
    weight_kg = processed["weight_kg"]
    goal_weight_kg = processed["goal_weight_kg"]
    goal_timeline_weeks = processed["goal_timeline_weeks"]
    days_per_week = processed["days_per_week"]
    sex_bin = processed["sex_bin"]
    bmi = processed["bmi"]
    goal_direction = processed["goal_direction"]
    lifestyle = processed
    
    predicted_body_type = classifier.predict([[age, height_cm, weight_kg, sex_bin]])[0]
    
    calorie_target = estimate_calories(
    age, sex, height_cm, weight_kg, goal_weight_kg, goal_timeline_weeks, days_per_week, activity_df
    )

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

    # --- k-NN Twin Retrieval (with StandardScaler + lifestyle weights) ---
    finder = DietTwinFinder(
        diet_df, metric="cosine",
        feature_cols=DIET_FEATURE_COLS, weights=DIET_FEATURE_WEIGHTS,
    )
    twin_indices, twin_distances = finder.find_twin(user_vector, k=1)
    twin = diet_df.iloc[twin_indices[0]]
    similarity = float(1.0 - twin_distances[0])

    # --- Macro Target Calculation (scientific, goal-based) ---
    if goal_direction == -1:  # Weight Loss
        target_protein = (calorie_target * 0.40) / 4
        target_carbs = (calorie_target * 0.30) / 4
        target_fat = (calorie_target * 0.30) / 9
    elif goal_direction == 1:  # Muscle Gain
        target_protein = (calorie_target * 0.30) / 4
        target_carbs = (calorie_target * 0.50) / 4
        target_fat = (calorie_target * 0.20) / 9
    else:  # Maintenance
        target_protein = (calorie_target * 0.30) / 4
        target_carbs = (calorie_target * 0.40) / 4
        target_fat = (calorie_target * 0.30) / 9

    # --- Monte Carlo Meal Generation ---
    generator = MealGenerator(food_df)
    meals, meal_error, actual_totals = generator.generate_meal_plan(
        calorie_target, target_protein, target_carbs, target_fat,
        num_meals=7, iterations=10000,
    )

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
        # Assign human-readable meal labels to the 7-dish plan
        meal_labels = [
            "Breakfast", "Breakfast",
            "Lunch", "Lunch", "Lunch",
            "Dinner", "Dinner",
        ]
        display_df = meals[["Name", "Calories", "Protein", "Carbs", "Fat"]].copy()
        display_df.insert(0, "Meal", meal_labels[:len(display_df)])
        display_df.columns = ["Meal", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
        st.markdown('<div class="table-shell">', unsafe_allow_html=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="small-note">
                Meals optimized via Monte Carlo simulation (10,000 iterations).
                Actual totals — Calories: <b>{actual_totals['Calories']:.0f}</b> |
                Protein: <b>{actual_totals['Protein']:.0f} g</b> |
                Carbs: <b>{actual_totals['Carbs']:.0f} g</b> |
                Fat: <b>{actual_totals['Fat']:.0f} g</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_workouts:
        st.markdown('<div class="section-title">Recommended Weekly Lift Plan</div>', unsafe_allow_html=True)
        workout_table = workouts[["exercise_name", "muscle_group", "difficulty", "equipment", "duration_min"]].copy()
        workout_table.columns = ["Exercise", "Muscle Group", "Difficulty", "Equipment", "Duration (min)"]
        st.markdown('<div class="table-shell">', unsafe_allow_html=True)
        st.dataframe(workout_table, use_container_width=True, hide_index=True)
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
                "- Update your lifestyle choices in Edit Profile for deeper personalization."
            )
        if lifestyle["injury_care"] or lifestyle["low_sleep"] or lifestyle["medical_condition"]:
            st.info("Recovery-sensitive mode is on: prioritize consistency, moderate intensity, and safe food/training choices.")


def render_onboarding_form(existing_profile=None):
    defaults = existing_profile or {}
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
        days_per_week = st.slider(
            "Workout days / week", 0, 7, int(defaults.get("days_per_week", 4))
        )
        schedule_type = st.selectbox(
            "What is your daily schedule like?",
            ["Regular daytime", "Night shift", "Irregular schedule"],
            index=["Regular daytime", "Night shift", "Irregular schedule"].index(defaults.get("schedule_type", "Regular daytime")),
        )
        workout_location = st.selectbox(
            "Where do you prefer to work out?",
            ["Gym", "Home", "Both"],
            index=["Gym", "Home", "Both"].index(defaults.get("workout_location", "Home")),
        )
        workout_time = st.selectbox(
            "How much time can you exercise per session?",
            ["15-20 minutes", "30-45 minutes", "60+ minutes"],
            index=["15-20 minutes", "30-45 minutes", "60+ minutes"].index(defaults.get("workout_time", "30-45 minutes")),
        )
        diet_preference = st.selectbox(
            "Diet preference",
            ["No preference", "Vegetarian", "High protein", "Low carb"],
            index=["No preference", "Vegetarian", "High protein", "Low carb"].index(defaults.get("diet_preference", "No preference")),
        )
        craving_level = st.selectbox(
            "Do you often crave sweets/snacks?",
            ["Rarely", "Sometimes", "Often"],
            index=["Rarely", "Sometimes", "Often"].index(defaults.get("craving_level", "Sometimes")),
        )
        stress_level = st.selectbox(
            "Stress level",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(defaults.get("stress_level", "Medium")),
        )
        sleep_quality = st.selectbox(
            "Sleep quality",
            ["Good", "Average", "Poor"],
            index=["Good", "Average", "Poor"].index(defaults.get("sleep_quality", "Average")),
        )
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
        default_conditions = defaults.get("health_conditions", ["None"])
        if isinstance(default_conditions, str):
            default_conditions = default_conditions.split(",")
        default_conditions = [c for c in default_conditions if c in health_options]
        if not default_conditions:
            default_conditions = ["None"]
        health_conditions = st.multiselect(
            "Do you have any injuries or health conditions? (Select all that apply)",
            health_options,
            default=default_conditions,
        )
        submitted = st.form_submit_button("Save and Generate Plan")
    if not submitted:
        return None
    if not health_conditions:
        health_conditions = ["None"]
    return {
        "age": age,
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "goal_weight_kg": goal_weight_kg,
        "goal_timeline_weeks": goal_timeline_weeks,
        "days_per_week": days_per_week,
        "schedule_type": schedule_type,
        "workout_location": workout_location,
        "workout_time": workout_time,
        "diet_preference": diet_preference,
        "craving_level": craving_level,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "health_conditions": health_conditions,
    }


def set_onboarding_sex(value: str) -> None:
    """Callback runs before the rest of the script; do not call st.rerun() here."""
    st.session_state["onboarding_selected_sex"] = value


def render_onboarding_wizard():
    steps = [
        ("age", "How old are you?"),
        ("sex", "What is your sex?"),
        ("height_cm", "What is your height (cm)?"),
        ("weight_kg", "What is your current weight (kg)?"),
        ("goal_weight_kg", "What is your goal weight (kg)?"),
        ("goal_timeline_weeks", "How many weeks do you expect to reach your goal?"),
        ("days_per_week", "How many days per week can you work out?"),
        ("schedule_type", "What is your daily schedule like?"),
        ("workout_location", "Where do you prefer to work out?"),
        ("workout_time", "How much time can you exercise per session?"),
        ("diet_preference", "What is your diet preference?"),
        ("craving_level", "Do you often crave sweets/snacks?"),
        ("stress_level", "What is your stress level?"),
        ("sleep_quality", "How is your sleep quality?"),
        ("health_conditions", "Do you have any health conditions?"),
    ]
    defaults = {
        "name": "",
        "age": 24,
        "sex": "M",
        "height_cm": 172,
        "weight_kg": 75,
        "goal_weight_kg": 68,
        "goal_timeline_weeks": 12,
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
        "goal_timeline_weeks": "T",
        "days_per_week": "D",
        "schedule_type": "ST",
        "workout_location": "WL",
        "workout_time": "WT",
        "diet_preference": "DP",
        "craving_level": "C",
        "stress_level": "SL",
        "sleep_quality": "SQ",
        "health_conditions": "HC",
    }
    days_to_activity = {v: k for k, v in activity_to_days.items()}
    params = st.query_params
    param_sex = str(params.get("onboarding_sex", "")).upper()
    param_height_raw = str(params.get("onboarding_height", "")).strip()
    param_next = str(params.get("onboarding_next", "0")).strip()
    param_back = str(params.get("onboarding_back", "0")).strip()
    if param_back == "1":
        st.session_state.pop("onboarding_selected_sex", None)
        st.session_state["onboarding_stage"] = "height"
        params.clear()
        st.rerun()
    if param_height_raw.isdigit():
        parsed_height = int(param_height_raw)
        if 145 <= parsed_height <= 230:
            st.session_state["onboarding_height_cm"] = parsed_height
    if param_next == "1":
        st.session_state["onboarding_stage"] = "details"
        params.clear()
        st.rerun()
    if param_sex in {"M", "F"}:
        st.session_state["onboarding_selected_sex"] = param_sex
        params.clear()
        st.rerun()
    selected_sex = st.session_state.get("onboarding_selected_sex")
    show_gender_page = selected_sex is None

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
                padding: 4px 20px 22px 20px;
            }
            .assessment-card {
                background: #ffffff;
                border-radius: 34px;
                border: none;
                box-shadow: 0 18px 50px rgba(20, 90, 70, 0.1);
                padding: 1.35rem 1.25rem 1.45rem 1.25rem;
            }
            body:has(.assessment-gender-step) .assessment-card.assessment-gender-step {
                margin-top: -10px;
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
                margin-top: -8px !important;
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
                transform: translateY(-52px);
            }
            .height-step-shell {
                background: transparent;
                border-radius: 0;
                box-shadow: none;
                padding: 0.05rem 0.9rem 0.6rem 0.9rem;
            }
            .height-top-back-wrap {
                display: flex;
                justify-content: flex-start;
                margin: -118px 0 44px 4px;
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
                min-height: 520px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                margin-bottom: 0.8rem;
            }
            .height-ruler-column {
                width: 94px;
                height: 430px;
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
            .height-ruler-mark.top { top: 38px; }
            .height-ruler-mark.mid { top: 178px; }
            .height-ruler-mark.bot { top: 318px; }
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
            body:has(.height-stage-lock),
            body:has(.height-stage-lock) html,
            body:has(.height-stage-lock) [data-testid="stAppViewContainer"],
            body:has(.height-stage-lock) [data-testid="stAppViewContainer"] > .main,
            body:has(.height-stage-lock) .main .block-container {
                height: 100svh !important;
                max-height: 100svh !important;
                overflow: hidden !important;
                overscroll-behavior: none !important;
            }
            body:has(.height-stage-lock) .main .block-container {
                padding-top: 0.1rem !important;
                padding-bottom: 0 !important;
            }
        </style>
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
        elif field == "goal_timeline_weeks":
            value = st.slider("How many weeks do you expect to reach your goal?", 4, 52, int(profile["goal_timeline_weeks"]))
        elif field == "days_per_week":
            value = st.slider("Workout days / week", 0, 7, int(profile["days_per_week"]))
        elif field == "schedule_type":
            value = st.selectbox(
                "What is your daily schedule like?",
                ["Regular daytime", "Night shift", "Irregular schedule"],
                index=["Regular daytime", "Night shift", "Irregular schedule"].index(profile["schedule_type"]),
            )
        elif field == "workout_location":
            value = st.selectbox(
                "Where do you prefer to work out?",
                ["Gym", "Home", "Both"],
                index=["Gym", "Home", "Both"].index(profile["workout_location"]),
            )
        elif field == "workout_time":
            value = st.selectbox(
                "How much time can you exercise per session?",
                ["15-20 minutes", "30-45 minutes", "60+ minutes"],
                index=["15-20 minutes", "30-45 minutes", "60+ minutes"].index(profile["workout_time"]),
            )
        elif field == "diet_preference":
            value = st.selectbox(
                "Diet preference",
                ["No preference", "Vegetarian", "High protein", "Low carb"],
                index=["No preference", "Vegetarian", "High protein", "Low carb"].index(profile["diet_preference"]),
            )
        elif field == "craving_level":
            value = st.selectbox(
                "Do you often crave sweets/snacks?",
                ["Rarely", "Sometimes", "Often"],
                index=["Rarely", "Sometimes", "Often"].index(profile["craving_level"]),
            )
        elif field == "stress_level":
            value = st.selectbox(
                "Stress level",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(profile["stress_level"]),
            )
        elif field == "sleep_quality":
            value = st.selectbox(
                "Sleep quality",
                ["Good", "Average", "Poor"],
                index=["Good", "Average", "Poor"].index(profile["sleep_quality"]),
            )
        elif field == "health_conditions":
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
            default_conditions = profile.get("health_conditions", ["None"])
            if isinstance(default_conditions, str):
                default_conditions = default_conditions.split(",")
            default_conditions = [c for c in default_conditions if c in health_options]
            if not default_conditions:
                default_conditions = ["None"]
            value = st.multiselect(
                "Do you have any of the following conditions? (Select all that apply)",
                health_options,
                default=default_conditions,
            )
        c1, c2 = st.columns([1, 1])
        back_clicked = c1.form_submit_button("Back", disabled=current_step == 0, type="secondary")
        next_label = "Save and Generate Plan" if current_step == len(steps) - 1 else "Next"
        next_clicked = c2.form_submit_button(next_label, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if back_clicked and current_step > 0:
        st.session_state["onboarding_step"] = current_step - 1
        st.rerun()

    if show_gender_page:
        male_path, female_path = resolve_gender_avatar_paths()
        st.markdown('<div class="assessment-shell">', unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent(
                """
                <div class="assessment-card assessment-gender-step">
                    <div class="assessment-progress"><div class="assessment-progress-fill"></div></div>
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

        final_profile = st.session_state["onboarding_profile"].copy()
        if not final_profile.get("health_conditions"):
            final_profile["health_conditions"] = ["None"]
        bmi = final_profile["weight_kg"] / ((final_profile["height_cm"] / 100) ** 2)
        st.info(f"Captured BMI: **{bmi:.1f}**")
        st.session_state.pop("onboarding_step", None)
        st.session_state.pop("onboarding_profile", None)
        return final_profile

    if "onboarding_stage" not in st.session_state:
        st.session_state["onboarding_stage"] = "height"
    if "onboarding_height_cm" not in st.session_state:
        st.session_state["onboarding_height_cm"] = int(defaults["height_cm"])

    if st.session_state["onboarding_stage"] == "height":
        current_height = int(st.session_state.get("onboarding_height_cm", defaults["height_cm"]))

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
            "img/test_img.png",
            use_container_width=True,
        )
    with right:
        st.markdown('<div class="fit-auth-right">', unsafe_allow_html=True)
        st.markdown('<div class="fit-auth-kicker">HI, WELCOME TO DIET TWIN COACHING</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="height-top-back-wrap"><a class="height-top-back-link" href="?onboarding_back=1" target="_self" rel="noopener">Back</a></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="assessment-shell" style="padding:4px 20px 10px 20px;">', unsafe_allow_html=True)
        st.markdown(
            """
                <div class="assessment-card" style="margin-top:-44px; margin-bottom:10px;">
                    <div class="assessment-progress"><div class="assessment-progress-fill" style="width:14%;"></div></div>
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
                    min-height: 300px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 20px;
                }}
                .col {{
                    position: relative;
                    width: 94px;
                    height: 430px;
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
                    top: 210px;
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
                <div class="next-wrap">
                    <button id="nextBtn" class="next-btn" type="button">Next</button>
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
                const centerY = 215;
                const rulerHeight = 430;
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
                }}

                function commitSelectionAndNext() {{
                    const url = `?onboarding_sex={selected_sex_for_link}&onboarding_height=${{current}}&onboarding_next=1`;
                    window.top.location.assign(url);
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
                // Prevent wheel scroll from bubbling to parent page.
                window.addEventListener("wheel", (e) => {{
                    e.preventDefault();
                }}, {{ passive: false }});

                document.getElementById("nextBtn").addEventListener("click", commitSelectionAndNext);
                buildTrack();
                render(current);
            </script>
        </body>
        </html>
        """
        components.html(ruler_component_html, height=540, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)  # .height-step-shell
        st.markdown("</div>", unsafe_allow_html=True)  # .basic-info-wrap
        return None

    st.markdown('<div class="basic-info-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="basic-info-title">Basic Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="basic-info-subtitle">Help us build your plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="basic-info-card">', unsafe_allow_html=True)

    with st.form("basic_information_form"):
        name = st.text_input("Name", value=defaults["name"], placeholder="Your name")
        age = st.number_input("Age", min_value=16, max_value=75, value=defaults["age"], step=1)
        gender_label = st.selectbox("Gender", ["Male", "Female"], index=0 if selected_sex == "M" else 1)
        height_cm = st.number_input(
            "Height (cm)",
            min_value=145,
            max_value=230,
            value=int(st.session_state.get("onboarding_height_cm", defaults["height_cm"])),
            step=1,
        )
        weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=250.0, value=float(defaults["weight_kg"]), step=0.1)
        goal_weight_kg = st.number_input("Goal (kg)", min_value=35.0, max_value=250.0, value=float(defaults["goal_weight_kg"]), step=0.1)
        activity_default = days_to_activity.get(defaults["days_per_week"], "Moderately Active")
        activity_level = st.selectbox(
            "Activity Level",
            activity_options,
            index=activity_options.index(activity_default),
        )
        back_to_gender = st.form_submit_button("Back", type="secondary", use_container_width=True)
        submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if back_to_gender:
        st.session_state.pop("onboarding_selected_sex", None)
        st.session_state["onboarding_stage"] = "height"
        st.rerun()
    if not submitted:
        return None

    sex = "M" if gender_label == "Male" else "F"
    days_per_week = activity_to_days[activity_level]
    lifestyle_text = (
        f"{activity_level.lower()} routine with about {days_per_week} workout days each week. "
        "Wants a practical, consistent plan that is easy to follow."
    )
    st.session_state["onboarding_stage"] = "height"
    st.session_state.pop("onboarding_selected_sex", None)
    return {
        "name": name.strip(),
        "age": int(age),
        "sex": sex,
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "goal_weight_kg": float(goal_weight_kg),
        "days_per_week": int(days_per_week),
        "lifestyle_text": lifestyle_text,
    }


def main():
    st.set_page_config(page_title="Diet Twin Planner", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_theme()
    params = st.query_params
    if str(params.get("onboarding_back", "0")).strip() == "1":
        # Back-to-gender navigation should never bounce to welcome page.
        st.session_state["welcome_seen"] = True
    if str(params.get("onboarding_sex", "")).upper() in {"M", "F"}:
        # If the app hard-reloads from an onboarding gender click, skip welcome.
        st.session_state["welcome_seen"] = True

    if "profile" not in st.session_state:
        st.session_state["profile"] = None
    if "welcome_seen" not in st.session_state:
        st.session_state["welcome_seen"] = False

    if not st.session_state["welcome_seen"]:
        if render_welcome_page():
            st.session_state["welcome_seen"] = True
            st.rerun()
        return

    body_df, diet_df, gym_df, food_df, activity_df = load_data()

    profile = st.session_state["profile"]
    if profile is None:
        new_profile = render_onboarding_wizard()
        if new_profile:
            st.session_state["profile"] = new_profile
            st.success("Profile saved. Building your first plan...")
            st.rerun()
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

    st.subheader("Your profile")
    with st.expander("Edit profile"):
        updated = render_onboarding_form(existing_profile=profile)
        if updated:
            st.session_state["profile"] = updated
            profile = st.session_state["profile"]
            st.success("Profile updated.")

    render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df)

if __name__ == "__main__":
    main()
