from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from diet_twin_finder import DietTwinFinder
from meal_generator import MealGenerator
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

LIFESTYLE_FIT_FEATURE_COLS = DIET_FEATURE_COLS


@st.cache_resource
def build_lifestyle_fit_model(diet_df: pd.DataFrame):
    """Train a Random Forest model that predicts plan adherence from lifestyle signals."""
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


def predict_lifestyle_fit(model_bundle: dict, user_vector: np.ndarray, lifestyle: dict):
    user_row = pd.DataFrame([user_vector], columns=LIFESTYLE_FIT_FEATURE_COLS)
    user_row = user_row.fillna(model_bundle["feature_medians"])

    regressor = model_bundle["regressor"]
    classifier = model_bundle["classifier"]
    user_array = user_row.to_numpy()
    tree_predictions = np.array([tree.predict(user_array)[0] for tree in regressor.estimators_])
    raw_score = float(np.clip(tree_predictions.mean(), 0, 100))
    uncertainty = float(tree_predictions.std())

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

    fit_score = float(np.clip(raw_score + score_adjustment, 0, 100))
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
        "label": label,
        "pattern": pattern.replace("_", " ").title(),
        "pattern_confidence": pattern_confidence,
        "uncertainty": uncertainty,
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
    return recommendations[:6]


def pick_workouts(
    gym_df: pd.DataFrame,
    home_workout: int,
    gym_workout: int,
    both_workout: int,
    short_sessions: int,
    days_per_week: int,
    health_conditions=None,
    workout_time="30-45 minutes",
):
    d = gym_df.copy()
    health_conditions = health_conditions or []
    conditions = [c.lower() for c in health_conditions]

    home_equipment = ["bodyweight", "dumbbell", "resistance_band", "bands"]
    gym_equipment = [
    "barbell", "machine", "cable", "bench",
    "dumbbell", "kettlebell", "bodyweight"
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

    # --- k-NN Twin Retrieval (with StandardScaler + lifestyle weights) ---
    finder = DietTwinFinder(
        diet_df, metric="cosine",
        feature_cols=DIET_FEATURE_COLS, weights=DIET_FEATURE_WEIGHTS,
    )
    twin_indices, twin_distances = finder.find_twin(user_vector, k=1)
    twin = diet_df.iloc[twin_indices[0]]
    similarity = float(1.0 - twin_distances[0])

    lifestyle_model = build_lifestyle_fit_model(diet_df)
    lifestyle_fit = predict_lifestyle_fit(lifestyle_model, user_vector, lifestyle)

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

    workouts = pick_workouts(
    gym_df,
    lifestyle["home_workout"],
    lifestyle["gym_workout"],
    lifestyle["both_workout"],
    lifestyle["short_sessions"],
    days_per_week,
    profile.get("health_conditions", []),
    profile.get("workout_time", "30-45 minutes"),
)
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
        st.markdown('<div class="section-title">Recommended Weekly Exercise Plan</div>', unsafe_allow_html=True)

        if workouts.empty:
            st.warning("No workout recommendations were found. Please update your profile and try again.")
        else:
            workout_table = workouts[
                ["exercise_name", "muscle_group", "difficulty", "equipment", "duration_min"]
            ].copy()

            workout_table.columns = [
                "Exercise",
                "Muscle Group",
                "Difficulty",
                "Equipment",
                "Duration (min)"
            ]

            workout_table.insert(0, "Day", [f"Day {i + 1}" for i in range(len(workout_table))])

            total_duration = workout_table["Duration (min)"].sum()
            avg_duration = workout_table["Duration (min)"].mean()
            equipment_needed = ", ".join(workout_table["Equipment"].dropna().unique())

            s1, s2, s3 = st.columns(3)
            s1.metric("Total Weekly Time", f"{total_duration:.0f} min")
            s2.metric("Average Session", f"{avg_duration:.0f} min")
            s3.metric("Equipment", equipment_needed if equipment_needed else "None")

            st.markdown("### Daily Workout Cards")

            for _, row in workout_table.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        background: white;
                        border-radius: 16px;
                        padding: 18px 22px;
                        margin-bottom: 14px;
                        box-shadow: rgba(0, 0, 0, 0.08) 0px 6px 20px;
                        border-left: 6px solid #0071e3;
                    ">
                        <div style="font-size: 0.9rem; color: #6e6e73; font-weight: 600;">
                            {row["Day"]}
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 700; margin-top: 4px;">
                            {row["Exercise"]}
                        </div>
                        <div style="margin-top: 10px; color: #333;">
                            <span style="
                                display:inline-block;
                                background:#f2f2f7;
                                padding:5px 10px;
                                border-radius:999px;
                                margin-right:6px;
                                font-size:0.9rem;
                            ">
                                Muscle: {row["Muscle Group"]}
                            </span>
                            <span style="
                                display:inline-block;
                                background:#f2f2f7;
                                padding:5px 10px;
                                border-radius:999px;
                                margin-right:6px;
                                font-size:0.9rem;
                            ">
                                Equipment: {row["Equipment"]}
                            </span>
                            <span style="
                                display:inline-block;
                                background:#f2f2f7;
                                padding:5px 10px;
                                border-radius:999px;
                                font-size:0.9rem;
                            ">
                                {row["Duration (min)"]} min
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("View workout table"):
                st.dataframe(workout_table, use_container_width=True, hide_index=True)

            if lifestyle["home_workout"]:
                st.info("This plan prioritizes home-friendly exercises based on your profile.")

            if lifestyle["short_sessions"]:
                st.info("This plan keeps sessions shorter because you selected limited workout time.")

            if lifestyle["injury_care"]:
                st.warning(
                    "Recovery-sensitive mode: because you reported an injury or pain condition, "
                    "choose moderate intensity and avoid movements that cause discomfort."
                )



    with tab_lifestyle:
        st.markdown('<div class="section-title">Why This Plan Fits Your Lifestyle</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        f1.metric("Lifestyle Fit", f"{lifestyle_fit['score']:.0f} / 100", lifestyle_fit["label"])
        f2.metric("Predicted Pattern", lifestyle_fit["pattern"])
        f3.metric("Model Confidence", f"{lifestyle_fit['pattern_confidence']:.0%}")

        st.markdown(
            f"""
            <div class="block-card">
                <b>Machine learning output:</b> A Random Forest model predicts that this plan has a
                <b>{lifestyle_fit['label'].lower()}</b> for your current routine. The raw model estimate was
                <b>{lifestyle_fit['raw_score']:.0f}%</b>, then the app applied small safety adjustments for
                sleep, injuries, and medical constraints that are not present in the historical training file.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cues = lifestyle["matched_cues"]
        if cues:
            st.markdown("- " + "\n- ".join([f"We adjusted recommendations because {c}." for c in cues[:5]]))
        else:
            st.markdown(
                "- We used your body profile and goal to personalize the baseline plan.\n"
                "- Update your lifestyle choices in Edit Profile for deeper personalization."
            )

        st.markdown("### Fit Recommendations")
        for recommendation in lifestyle_fit_recommendations(lifestyle, lifestyle_fit, calorie_target):
            st.markdown(f"- {recommendation}")

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
        st.dataframe(driver_df, use_container_width=True, hide_index=True)

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
