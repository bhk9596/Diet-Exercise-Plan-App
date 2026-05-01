from __future__ import annotations

from string import Formatter
from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_value(value: Any, default: str = "your") -> str:
    text = str(value).strip()
    return text if text else default


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def build_recommendation_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "default_meal_system",
            "category": "nutrition",
            "base_score": 2,
            "positive_signals": {
                "medium_stress": 3,
                "high_stress": 6,
                "low_fit_score": 4,
                "sugar_craving": 2,
                "twin_lower_adherence": 3,
            },
            "negative_signals": {
                "strong_fit_score": 2,
            },
            "template": (
                "Use a default meal system on busier days: keep {default_meal_count} repeatable meals "
                "available so your {calorie_target_text} target does not depend on last-minute decisions."
            ),
        },
        {
            "id": "planned_sweet_option",
            "category": "nutrition",
            "base_score": 0,
            "positive_signals": {
                "sugar_craving": 7,
                "twin_high_sugar_pattern": 5,
                "night_shift": 2,
                "low_fit_score": 2,
            },
            "negative_signals": {
                "rare_cravings": 5,
            },
            "template": (
                "Plan one controlled sweet or snack option after a protein-rich meal. This gives cravings "
                "a defined place in the day while keeping the full plan near {calorie_target_text}."
            ),
        },
        {
            "id": "night_shift_meal_timing",
            "category": "schedule",
            "base_score": 0,
            "positive_signals": {
                "night_shift": 9,
                "low_sleep": 3,
                "sugar_craving": 2,
                "twin_lower_adherence": 2,
            },
            "negative_signals": {
                "regular_schedule": 4,
            },
            "template": (
                "Anchor meals around your wake time instead of the clock: make the first main meal the "
                "most complete one, carry a planned protein snack, and keep the final meal lighter before sleep."
            ),
        },
        {
            "id": "short_session_training",
            "category": "training",
            "base_score": 0,
            "positive_signals": {
                "short_sessions": 8,
                "home_workout": 3,
                "low_fit_score": 3,
                "twin_lower_adherence": 2,
            },
            "negative_signals": {
                "long_sessions": 3,
                "injury_care": 1,
            },
            "template": (
                "Treat {session_length_text} training as the default, not the backup. Use {workout_days_text} "
                "focused sessions with one lower-body movement, one push or pull, and one core movement."
            ),
        },
        {
            "id": "home_workout_setup",
            "category": "training",
            "base_score": 0,
            "positive_signals": {
                "home_workout": 7,
                "short_sessions": 3,
                "medium_stress": 2,
                "high_stress": 3,
            },
            "negative_signals": {
                "gym_workout": 4,
            },
            "template": (
                "Make the {workout_location_text} setup frictionless: keep {equipment_text} ready in one visible "
                "spot and start each session with a short warmup before deciding whether to extend it."
            ),
        },
        {
            "id": "recovery_sensitive_training",
            "category": "recovery",
            "base_score": 0,
            "positive_signals": {
                "low_sleep": 7,
                "injury_care": 6,
                "medical_condition": 4,
                "high_stress": 3,
            },
            "negative_signals": {
                "strong_fit_score": 1,
            },
            "template": (
                "Use recovery-sensitive training rules: on poor-sleep or high-symptom days, keep the habit "
                "but lower intensity with walking, mobility, or lighter controlled sets."
            ),
        },
        {
            "id": "clinician_check",
            "category": "safety",
            "base_score": 0,
            "positive_signals": {
                "medical_condition": 10,
                "injury_care": 5,
            },
            "negative_signals": {},
            "template": (
                "Because health constraints are part of the profile, keep intensity moderate and confirm major "
                "diet or training changes with a qualified clinician."
            ),
        },
        {
            "id": "weekly_calendar_commitment",
            "category": "planning",
            "base_score": 2,
            "positive_signals": {
                "good_fit_score": 4,
                "strong_fit_score": 5,
                "low_stress": 3,
                "home_workout": 1,
            },
            "negative_signals": {
                "low_fit_score": 2,
            },
            "template": (
                "Use a weekly calendar commitment: schedule {workout_days_text} workouts before the week starts "
                "and choose the meals you will repeat before grocery decisions pile up."
            ),
        },
        {
            "id": "minimum_viable_week",
            "category": "adherence",
            "base_score": 0,
            "positive_signals": {
                "very_low_fit_score": 8,
                "low_fit_score": 5,
                "high_stress": 4,
                "twin_lower_adherence": 3,
                "low_sleep": 2,
            },
            "negative_signals": {
                "strong_fit_score": 5,
            },
            "template": (
                "For week one, use a minimum viable plan: protect one meal rule and the first 10 minutes "
                "of each workout before adding more complexity."
            ),
        },
        {
            "id": "protein_anchor",
            "category": "nutrition",
            "base_score": 1,
            "positive_signals": {
                "high_protein_preference": 5,
                "sugar_craving": 3,
                "low_fit_score": 2,
                "vegetarian_pref": 2,
            },
            "negative_signals": {},
            "template": (
                "Use protein as the anchor for each main meal. Your generated meals average about "
                "{meal_protein_text} protein per main meal, so keep that pattern when swapping foods."
            ),
        },
        {
            "id": "vegetarian_consistency",
            "category": "nutrition",
            "base_score": 0,
            "positive_signals": {
                "vegetarian_pref": 8,
                "high_protein_preference": 2,
                "low_fit_score": 2,
            },
            "negative_signals": {},
            "template": (
                "Keep vegetarian meals built around a clear protein source first, then add carbs and fats "
                "around it so the meal still supports the daily macro target."
            ),
        },
        {
            "id": "diet_twin_pattern",
            "category": "diet_twin",
            "base_score": 0,
            "positive_signals": {
                "strong_twin_match": 5,
                "twin_higher_adherence": 4,
                "twin_lower_adherence": 3,
            },
            "negative_signals": {
                "weak_twin_match": 4,
            },
            "template": (
                "Use the diet twin signal as a guardrail: the closest matched profile followed a "
                "{twin_pattern_text} pattern with {twin_adherence_text} adherence, so adjust gradually "
                "instead of rebuilding everything at once."
            ),
        },
    ]


def build_recommendation_context(
    profile: dict[str, Any],
    lifestyle: dict[str, Any],
    fit_result: dict[str, Any],
    twin: Any = None,
    meals: pd.DataFrame | None = None,
    workouts: pd.DataFrame | None = None,
    calorie_target: int | float | None = None,
) -> dict[str, Any]:
    fit_score = _safe_float(fit_result.get("score"), 0)
    twin_dict = twin.to_dict() if hasattr(twin, "to_dict") else dict(twin or {})
    twin_adherence = _safe_float(
        _first_present(twin_dict, ["adherence_score", "adherence"], None),
        default=-1,
    )
    twin_pattern = str(_first_present(twin_dict, ["diet_pattern", "pattern"], "mixed_balanced"))
    twin_similarity = _safe_float(fit_result.get("twin_influence"), 0) / 0.30
    twin_similarity = max(0.0, min(1.0, twin_similarity))

    days_per_week = int(_safe_float(profile.get("days_per_week"), 0))
    workout_days_text = f"{days_per_week} weekly" if days_per_week else "scheduled"
    calorie_target_text = f"{int(round(float(calorie_target)))}-calorie" if calorie_target else "daily calorie"
    session_length_text = _format_value(profile.get("workout_time"), "your available")
    workout_location_text = _format_value(profile.get("workout_location"), "training").lower()

    equipment_text = "basic equipment"
    if workouts is not None and not workouts.empty and "equipment" in workouts.columns:
        equipment_mode = workouts["equipment"].dropna().astype(str).str.strip()
        if not equipment_mode.empty:
            equipment_text = equipment_mode.mode().iat[0]

    meal_protein_text = "a steady amount of"
    if meals is not None and not meals.empty and {"meal_type", "protein_g"}.issubset(meals.columns):
        main_meals = meals[meals["meal_type"].isin(["breakfast", "lunch", "dinner"])]
        if not main_meals.empty:
            protein_by_meal = main_meals.groupby("meal_type")["protein_g"].sum()
            meal_protein_text = f"{protein_by_meal.mean():.0f} g of"

    craving_level = str(profile.get("craving_level", "")).strip().lower()
    schedule_type = str(profile.get("schedule_type", "")).strip().lower()
    workout_location = str(profile.get("workout_location", "")).strip().lower()
    diet_preference = str(profile.get("diet_preference", "")).strip().lower()

    signals = {
        "always": 1,
        "night_shift": int(bool(lifestyle.get("night_shift"))),
        "regular_schedule": int("regular" in schedule_type and not lifestyle.get("night_shift")),
        "sugar_craving": int(bool(lifestyle.get("sugar_craving"))),
        "rare_cravings": int(craving_level == "rarely"),
        "home_workout": int(bool(lifestyle.get("home_workout"))),
        "gym_workout": int(workout_location == "gym"),
        "vegetarian_pref": int(bool(lifestyle.get("vegetarian_pref"))),
        "high_protein_preference": int("high protein" in diet_preference),
        "low_stress": int(bool(lifestyle.get("low_stress"))),
        "medium_stress": int(bool(lifestyle.get("medium_stress"))),
        "high_stress": int(bool(lifestyle.get("high_stress"))),
        "short_sessions": int(bool(lifestyle.get("short_sessions"))),
        "long_sessions": int(bool(lifestyle.get("long_sessions"))),
        "low_sleep": int(bool(lifestyle.get("low_sleep"))),
        "injury_care": int(bool(lifestyle.get("injury_care"))),
        "medical_condition": int(bool(lifestyle.get("medical_condition"))),
        "very_low_fit_score": int(fit_score < 48),
        "low_fit_score": int(fit_score < 62),
        "good_fit_score": int(62 <= fit_score < 78),
        "strong_fit_score": int(fit_score >= 78),
        "twin_lower_adherence": int(0 <= twin_adherence < 62),
        "twin_higher_adherence": int(twin_adherence >= 78),
        "strong_twin_match": int(twin_similarity >= 0.70),
        "weak_twin_match": int(twin_similarity < 0.35),
        "twin_high_sugar_pattern": int("sugar" in twin_pattern.lower()),
    }

    return {
        "signals": signals,
        "calorie_target_text": calorie_target_text,
        "default_meal_count": 2 if signals["high_stress"] or signals["low_fit_score"] else 3,
        "session_length_text": session_length_text,
        "workout_days_text": workout_days_text,
        "workout_location_text": workout_location_text,
        "equipment_text": equipment_text,
        "meal_protein_text": meal_protein_text,
        "twin_pattern_text": twin_pattern.replace("_", " "),
        "twin_adherence_text": f"{twin_adherence:.0f}%" if twin_adherence >= 0 else "unknown",
    }


def score_recommendation(candidate: dict[str, Any], context: dict[str, Any]) -> float:
    signals = context["signals"]
    score = float(candidate.get("base_score", 0))
    for signal_name, weight in candidate.get("positive_signals", {}).items():
        score += float(weight) * float(signals.get(signal_name, 0))
    for signal_name, weight in candidate.get("negative_signals", {}).items():
        score -= float(weight) * float(signals.get(signal_name, 0))
    return score


def render_recommendation(candidate: dict[str, Any], context: dict[str, Any]) -> str:
    template = str(candidate["template"])
    field_names = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]
    values = {field_name: context.get(field_name, "") for field_name in field_names}
    return template.format(**values)


def generate_lifestyle_recommendations(
    profile: dict[str, Any],
    lifestyle: dict[str, Any],
    fit_result: dict[str, Any],
    twin: Any = None,
    meals: pd.DataFrame | None = None,
    workouts: pd.DataFrame | None = None,
    calorie_target: int | float | None = None,
    limit: int = 6,
) -> list[str]:
    context = build_recommendation_context(
        profile=profile,
        lifestyle=lifestyle,
        fit_result=fit_result,
        twin=twin,
        meals=meals,
        workouts=workouts,
        calorie_target=calorie_target,
    )

    scored = []
    for candidate in build_recommendation_catalog():
        score = score_recommendation(candidate, context)
        if score > 0:
            scored.append((score, candidate["id"], candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [render_recommendation(candidate, context) for _, _, candidate in scored[:limit]]
