from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from ui_sections import (
    apply_custom_theme_styles,
    render_onboarding_wizard,
    render_plan_screen,
    render_profile_form_ui,
    render_welcome_page,
)


DATA_DIR = Path(__file__).parent / "data"

LIFESTYLE_FIT_FEATURE_COLS = [
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


def parse_lifestyle(profile: dict):
    text = str(profile.get("lifestyle_text", ""))
    t = text.lower()
    schedule_type = profile.get("schedule_type", "Regular daytime")
    workout_location = profile.get("workout_location", "Home")
    workout_time = profile.get("workout_time", "30-45 minutes")
    diet_preference = profile.get("diet_preference", "No preference")
    craving_level = profile.get("craving_level", "Sometimes")
    stress_level = profile.get("stress_level", "Medium")
    sleep_quality = profile.get("sleep_quality", "Average")
    health_conditions = profile.get("health_conditions", ["None"])
    if isinstance(health_conditions, str):
        health_conditions = [x.strip() for x in health_conditions.split(",") if x.strip()]
    tags = {
        "night_shift": int(schedule_type == "Night shift" or any(k in t for k in ["night shift", "overnight", "late shift"])),
        "sugar_craving": int(craving_level == "Often" or any(k in t for k in ["sugar", "dessert", "sweet", "crave", "snack at night"])),
        "home_workout": int(workout_location == "Home" or any(k in t for k in ["home", "apartment", "no gym"])),
        "vegetarian_pref": int(diet_preference == "Vegetarian" or any(k in t for k in ["vegetarian", "plant-based", "vegan"])),
        "high_stress": int(stress_level == "High" or any(k in t for k in ["stress", "busy", "anxious", "burnout", "overwhelmed"])),
        "short_sessions": int(workout_time == "15-20 minutes" or any(k in t for k in ["20 minutes", "15 minutes", "short workout", "quick workout"])),
        "long_sessions": int(workout_time == "60+ minutes" or any(k in t for k in ["60 minutes", "60+ minutes", "long workout"])),
        "low_sleep": int(sleep_quality == "Poor" or any(k in t for k in ["sleep 5", "sleep 4", "insomnia", "poor sleep"])),
        "injury_care": int(
            any(c in health_conditions for c in ["Knee pain", "Back pain", "Shoulder injury", "Joint pain", "Severe Arthritis"])
            or any(k in t for k in ["injury", "knee pain", "back pain", "joint pain"])
        ),
        "medical_condition": int("None" not in health_conditions),
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
    if tags["medical_condition"]:
        matched_cues.append("you reported a medical condition")
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


@st.cache_resource
def build_lifestyle_fit_model(diet_df: pd.DataFrame):
    """Train Random Forest models that estimate lifestyle fit from profile signals."""
    model_df = diet_df.dropna(subset=["adherence_score", "diet_pattern"]).copy()
    x = model_df[LIFESTYLE_FIT_FEATURE_COLS].fillna(model_df[LIFESTYLE_FIT_FEATURE_COLS].median())
    y_score = model_df["adherence_score"].clip(0, 100)
    y_pattern = model_df["diet_pattern"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_score, test_size=0.2, random_state=42
    )

    regressor = RandomForestRegressor(
        n_estimators=260,
        max_depth=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=1,
    )
    regressor.fit(x_train, y_train)

    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(x, y_pattern)

    predictions = regressor.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    importances = pd.Series(
        regressor.feature_importances_,
        index=LIFESTYLE_FIT_FEATURE_COLS,
    ).sort_values(ascending=False)

    return {
        "regressor": regressor,
        "classifier": classifier,
        "feature_medians": x.median(),
        "metrics": metrics,
        "importances": importances,
    }


def predict_lifestyle_fit(
    model_bundle: dict,
    user_vector: np.ndarray,
    lifestyle: dict,
    twin_adherence_score: float | None = None,
    twin_similarity: float = 0.0,
):
    user_row = pd.DataFrame([user_vector], columns=LIFESTYLE_FIT_FEATURE_COLS)
    user_row = user_row.fillna(model_bundle["feature_medians"])

    regressor = model_bundle["regressor"]
    classifier = model_bundle["classifier"]
    user_array = user_row.to_numpy()
    tree_predictions = np.array([tree.predict(user_array)[0] for tree in regressor.estimators_])
    raw_score = float(np.clip(tree_predictions.mean(), 0, 100))

    pattern = str(classifier.predict(user_row)[0])
    pattern_probs = classifier.predict_proba(user_row)[0]
    pattern_confidence = float(pattern_probs.max())

    score_adjustment = 0
    if lifestyle["low_sleep"]:
        score_adjustment -= 6
    if lifestyle["injury_care"]:
        score_adjustment -= 5
    if lifestyle["medical_condition"]:
        score_adjustment -= 4
    if lifestyle["short_sessions"]:
        score_adjustment += 3
    if lifestyle["home_workout"]:
        score_adjustment += 2

    adjusted_score = float(np.clip(raw_score + score_adjustment, 0, 100))
    if twin_adherence_score is None or pd.isna(twin_adherence_score):
        fit_score = adjusted_score
        twin_weight = 0.0
        twin_adherence = None
    else:
        twin_adherence = float(np.clip(twin_adherence_score, 0, 100))
        twin_weight = float(np.clip(twin_similarity, 0, 1)) * 0.30
        fit_score = float(np.clip((adjusted_score * (1 - twin_weight)) + (twin_adherence * twin_weight), 0, 100))

    if fit_score >= 78:
        label = "Strong fit"
    elif fit_score >= 62:
        label = "Good fit with a few friction points"
    elif fit_score >= 48:
        label = "Moderate fit"
    else:
        label = "Needs lifestyle support"

    return {
        "score": fit_score,
        "raw_score": raw_score,
        "adjusted_score": adjusted_score,
        "label": label,
        "pattern": pattern.replace("_", " ").title(),
        "pattern_confidence": pattern_confidence,
        "twin_adherence_score": twin_adherence,
        "twin_influence": twin_weight,
        "top_features": model_bundle["importances"].head(4),
        "metrics": model_bundle["metrics"],
    }


def lifestyle_fit_recommendations(lifestyle: dict, fit_result: dict, calorie_target=None):
    recommendations = []
    daily_calorie_text = f" Keep the full day near {calorie_target} calories." if calorie_target else ""
    if lifestyle["night_shift"]:
        recommendations.append(
            "Night shift schedule: eat your largest meal within 1-2 hours after waking, pack one planned high-protein snack for the middle of your shift, and make the final meal lighter so it is easier to sleep after work."
        )
    if lifestyle["sugar_craving"]:
        recommendations.append(
            "Sweet/snack cravings: pre-plan one controlled sweet option per day, such as Greek yogurt with fruit, a protein bar, or oatmeal with cinnamon. Eat it after a protein-rich meal instead of grazing between meals."
        )
    if lifestyle["high_stress"]:
        recommendations.append(
            "High-stress days: use a simple backup menu instead of improvising. Pick two repeatable meals, such as chicken/rice/vegetables or eggs/toast/fruit, and keep them ready for days when decision-making is low."
        )
    if lifestyle["short_sessions"]:
        recommendations.append(
            "Short workout sessions: use a 15-20 minute circuit with 3 rounds of 4 movements: one lower-body move, one push, one pull, and one core move. Rest 30-45 seconds between movements so the session stays short."
        )
    if lifestyle["home_workout"]:
        recommendations.append(
            "Home workout setup: keep the plan limited to bodyweight, dumbbells, or resistance bands. Put equipment in one visible spot and start with a 3-minute warmup so setup time does not become the reason to skip."
        )
    if lifestyle["low_sleep"]:
        recommendations.append(
            "Poor sleep: if you slept badly, switch that day's workout to an easy walk, mobility, or one light set of each planned exercise. Keep the habit, but avoid max-effort lifting or high-intensity intervals."
        )
    if lifestyle["injury_care"] or lifestyle["medical_condition"]:
        recommendations.append(
            "Injury or medical constraint: avoid exercises that trigger pain, keep intensity moderate, and choose controlled movements over jumping or heavy loading. For medical conditions, confirm major diet or training changes with a qualified clinician."
        )
    if not recommendations:
        recommendations.append(
            f"Low-friction routine: focus on three basics each day: hit the calorie target, include protein at every main meal, and complete the scheduled workout days.{daily_calorie_text}"
        )

    if fit_result["score"] < 62:
        recommendations.append(
            "Lower predicted adherence: for week 1, do not change everything at once. Choose one meal rule, such as protein at breakfast, and one exercise rule, such as completing the first 10 minutes of each workout."
        )
    else:
        recommendations.append(
            "Weekly execution target: review the plan every Sunday, choose the meals you will repeat, and schedule workouts on your calendar before the week starts."
        )
    if fit_result.get("twin_adherence_score") is not None:
        recommendations.append(
            f"Diet twin signal: your closest matched profile averaged {fit_result['twin_adherence_score']:.0f}% adherence, so the lifestyle fit score is partially anchored to that real matched outcome."
        )
    return recommendations[:6]


def retrieve_diet_twin(user_vector: np.ndarray, diet_df: pd.DataFrame):
    # Higher weights on lifestyle features to emphasize behavior-fit over pure body stats.
    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.3, 2.6, 2.2, 1.8, 2.2, 2.2, 1.5], dtype=float)
    weighted_matrix = diet_df[LIFESTYLE_FIT_FEATURE_COLS].values * weights
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


def pick_workouts(
    gym_df: pd.DataFrame,
    workout_location: str,
    short_sessions: int,
    long_sessions: int,
    days_per_week: int,
    injury_care: int = 0,
    health_conditions=None,
):
    d = gym_df.copy()

    workout_location = str(workout_location).lower().strip()

    home_equipment = ["bodyweight", "dumbbell", "resistance_band", "bands"]
    gym_equipment = ["barbell", "machine", "cable", "other", "dumbbell", "bands"]

    if health_conditions is None:
        health_conditions = []

    health_text = " ".join([str(x).lower().strip() for x in health_conditions])

    avoid_keywords = []

    knee_keywords = [
        "knee", "squat", "lunge", "jump", "run", "running",
        "step", "step-up", "step up", "box jump", "burpee",
        "mountain climber", "high knees", "leg press",
        "leg extension", "leg curl", "calf", "calves",
        "ankle", "quad", "quadriceps", "hamstring",
        "glute", "glutes", "hip thrust", "hip lift",
        "bridge", "good morning", "side leg", "leg raise",
        "groiner", "kneeling", "kneel"
    ]

    back_keywords = [
        "back", "lower back", "spine", "deadlift",
        "good morning", "row", "bent-over", "bent over",
        "superman", "hyperextension", "extension",
        "crunch", "sit up", "sit-up", "twist",
        "rotation", "russian twist", "plank"
    ]

    shoulder_keywords = [
        "shoulder", "overhead", "press", "push up", "push-up",
        "dip", "dips", "raise", "lateral raise", "front raise",
        "upright row", "row", "fly", "arm circle",
        "bench press", "chest press", "plank"
    ]

    joint_keywords = [
        "jump", "run", "running", "burpee", "squat",
        "lunge", "step", "mountain climber", "high knees",
        "push up", "push-up", "dip", "press",
        "twist", "rotation", "kneeling", "kneel"
    ]

    if "knee" in health_text:
        avoid_keywords += knee_keywords

    if "back" in health_text:
        avoid_keywords += back_keywords

    if "shoulder" in health_text:
        avoid_keywords += shoulder_keywords

    if "joint" in health_text or "arthritis" in health_text:
        avoid_keywords += joint_keywords
    safe_base = d.copy()

    if avoid_keywords:
        pattern = "|".join(re.escape(k) for k in set(avoid_keywords))

        text_cols = ["exercise_name", "muscle_group", "equipment"]
        combined_text = safe_base[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

        safe_base = safe_base[~combined_text.str.contains(pattern, na=False, regex=True)]

    if injury_care:
        safe_base = safe_base[safe_base["difficulty"] <= 2]
    if safe_base.empty:
        safe_base = gym_df.copy()
        if injury_care:
            safe_base = safe_base[safe_base["difficulty"] <= 2]

    d = safe_base.copy()
    if workout_location == "home":
        loc_d = d[d["equipment"].isin(home_equipment)]
        if not loc_d.empty:
            d = loc_d

    elif workout_location == "gym":
        loc_d = d[d["equipment"].isin(gym_equipment)]
        if not loc_d.empty:
            d = loc_d

    elif workout_location == "both":
        loc_d = d[d["equipment"].isin(list(set(home_equipment + gym_equipment)))]
        if not loc_d.empty:
            d = loc_d
    if short_sessions:
        time_d = d[d["duration_min"] <= 25]
        if not time_d.empty:
            d = time_d

    elif long_sessions:
        time_d = d[d["duration_min"] >= 45]
        if not time_d.empty:
            d = time_d
        else:
            time_d = d[d["duration_min"] >= 30]
            if not time_d.empty:
                d = time_d

    else:
        time_d = d[(d["duration_min"] >= 30) & (d["duration_min"] <= 45)]
        if not time_d.empty:
            d = time_d
    if d.empty:
        st.warning("No exact workouts found for your selected preferences. Showing closest safe available exercises.")
        d = safe_base.copy()

    weekly_count = min(max(days_per_week, 3), 6)

    return (
        d.sort_values(by=["difficulty", "duration_min"])
        .drop_duplicates(subset=["exercise_name"])
        .head(weekly_count)
    )



def render_plan(profile, body_df, diet_df, gym_df, food_df, activity_df):
    classifier = build_body_classifier(body_df)

    age = int(profile["age"])
    sex = profile["sex"]
    height_cm = float(profile["height_cm"])
    weight_kg = float(profile["weight_kg"])
    goal_weight_kg = float(profile["goal_weight_kg"])
    days_per_week = int(profile["days_per_week"])

    lifestyle = parse_lifestyle(profile)
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
    lifestyle_model = build_lifestyle_fit_model(diet_df)
    twin_adherence_score = float(twin.get("adherence_score", np.nan))
    lifestyle_fit = predict_lifestyle_fit(
        lifestyle_model,
        user_vector,
        lifestyle,
        twin_adherence_score=twin_adherence_score,
        twin_similarity=similarity,
    )
    lifestyle_recommendations = lifestyle_fit_recommendations(lifestyle, lifestyle_fit, calorie_target)
    meals = pick_meals(food_df, calorie_target, lifestyle["vegetarian_pref"])
    workouts = pick_workouts(
    gym_df,
    profile.get("workout_location", "Home"),
    lifestyle["short_sessions"],
    lifestyle["long_sessions"],
    days_per_week,
    lifestyle["injury_care"],
    profile.get("health_conditions", []),
)
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
        "lifestyle_fit": lifestyle_fit,
        "lifestyle_recommendations": lifestyle_recommendations,
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
