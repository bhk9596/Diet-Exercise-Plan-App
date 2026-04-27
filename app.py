from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from ui_sections import (
    apply_custom_theme_styles,
    render_onboarding_wizard,
    render_plan_screen,
    render_profile_form_ui,
    render_welcome_page,
)


DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    body_df = pd.read_csv(DATA_DIR / "nhanes_body_profiles.csv")
    diet_df = pd.read_csv(DATA_DIR / "diet_lifestyle_profiles.csv")
    gym_df = pd.read_csv(DATA_DIR / "megagym_subset.csv")
    food_path = DATA_DIR / "food_catalog.csv"
    if not food_path.exists():
        food_path = DATA_DIR / "clean_food_catalog.csv"
    food_df = pd.read_csv(food_path)
    # Normalize food schema so both food_catalog.csv and clean_food_catalog.csv work.
    canonical_food_cols = {
        "name": "food_name",
        "food_name": "food_name",
        "calories": "calories",
        "calorie": "calories",
        "protein": "protein_g",
        "protein_g": "protein_g",
        "carbs": "carbs_g",
        "carb": "carbs_g",
        "carbs_g": "carbs_g",
        "fat": "fat_g",
        "fat_g": "fat_g",
        "vegetarian": "vegetarian",
        "is_vegetarian": "vegetarian",
    }
    rename_map = {}
    for col in food_df.columns:
        key = str(col).strip().lower()
        if key in canonical_food_cols:
            rename_map[col] = canonical_food_cols[key]
    if rename_map:
        food_df = food_df.rename(columns=rename_map)
    if "food_name" not in food_df.columns and "Name" in food_df.columns:
        food_df["food_name"] = food_df["Name"]
    for numeric_col in ("calories", "protein_g", "carbs_g", "fat_g"):
        if numeric_col not in food_df.columns:
            food_df[numeric_col] = 0.0
        food_df[numeric_col] = pd.to_numeric(food_df[numeric_col], errors="coerce").fillna(0.0)
    if "vegetarian" not in food_df.columns:
        food_df["vegetarian"] = 0
    food_df["vegetarian"] = pd.to_numeric(food_df["vegetarian"], errors="coerce").fillna(0).astype(int)
    activity_df = pd.read_csv(DATA_DIR / "activity_multipliers.csv")
    return body_df, diet_df, gym_df, food_df, activity_df


def apply_custom_theme():
    apply_custom_theme_styles()


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
        vegetarian_only = candidates[candidates["vegetarian"] == 1]
        if not vegetarian_only.empty:
            candidates = vegetarian_only
    candidates = candidates.sort_values(by="protein_g", ascending=False)
    if candidates.empty:
        return pd.DataFrame(columns=["meal_type", "food_name", "calories", "protein_g", "carbs_g", "fat_g"])
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

    return {
        "predicted_body_type": predicted_body_type,
        "bmi": bmi,
        "calorie_target": calorie_target,
        "similarity": similarity,
        "weight_kg": weight_kg,
        "goal_weight_kg": goal_weight_kg,
        "days_per_week": days_per_week,
        "goal_progress": goal_progress,
        "twin": twin,
        "meals": meals,
        "workouts": workouts,
        "lifestyle": lifestyle,
    }


def render_onboarding_form(existing_profile=None):
    defaults = existing_profile or {}
    return render_profile_form_ui(defaults)


def main():
    st.set_page_config(page_title="Diet Twin Planner", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_theme()
    params = st.query_params
    onboarding_param_keys = {
        "onboarding_back",
        "onboarding_sex",
        "onboarding_height",
        "onboarding_next",
        "onboarding_weight",
        "onboarding_goal_weight",
        "onboarding_goal_month",
        "onboarding_goal_day",
        "onboarding_goal_year",
        "onboarding_weight_back",
        "onboarding_age_back",
        "onboarding_goal_back",
        "onboarding_goal_time_back",
        "onboarding_plan_intro_back",
        "onboarding_details_back",
        "onboarding_birth_month",
        "onboarding_birth_day",
        "onboarding_birth_year",
        "onboarding_age_next",
    }
    if any(str(params.get(k, "")).strip() for k in onboarding_param_keys):
        # Any onboarding navigation param means user is already past welcome.
        st.session_state["welcome_seen"] = True
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

    plan_data = render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df)
    render_plan_screen(plan_data)

if __name__ == "__main__":
    main()
