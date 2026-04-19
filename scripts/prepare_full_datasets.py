from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _extract_total_goal(payload: str) -> tuple[dict[str, float], dict[str, float]]:
    try:
        parsed = json.loads(payload)
    except Exception:
        return {}, {}
    totals = {}
    goals = {}
    for key, dest in [("total", totals), ("goal", goals)]:
        values = parsed.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            name = str(item.get("name", "")).strip().lower()
            value = _safe_float(item.get("value"))
            if name and value is not None:
                dest[name] = value
    return totals, goals


def load_nhanes_profiles(demo_path: Path, bmx_path: Path) -> pd.DataFrame:
    demo = pd.read_sas(demo_path)
    bmx = pd.read_sas(bmx_path)

    merged = demo[["SEQN", "RIAGENDR", "RIDAGEYR"]].merge(
        bmx[["SEQN", "BMXHT", "BMXWT", "BMXBMI"]],
        on="SEQN",
        how="inner",
    )
    merged = merged.rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "sex_code",
            "BMXHT": "height_cm",
            "BMXWT": "weight_kg",
            "BMXBMI": "bmi",
        }
    )
    merged = merged.dropna(subset=["age", "sex_code", "height_cm", "weight_kg", "bmi"])
    merged["sex_bin"] = (merged["sex_code"] == 1).astype(int)

    bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    merged["body_type"] = pd.cut(merged["bmi"], bins=bins, labels=labels, right=False)

    profiles = merged[["age", "height_cm", "weight_kg", "sex_bin", "body_type"]].copy()
    profiles["age"] = profiles["age"].astype(int)
    profiles["height_cm"] = profiles["height_cm"].round(1)
    profiles["weight_kg"] = profiles["weight_kg"].round(1)
    profiles["body_type"] = profiles["body_type"].astype(str)
    return profiles


def load_mfp_aggregates(mfp_tsv_path: Path, max_rows: int) -> pd.DataFrame:
    column_names = ["user_id", "entry_date", "meals_json", "totals_json"]
    chunks = pd.read_csv(
        mfp_tsv_path,
        sep="\t",
        header=None,
        names=column_names,
        usecols=[0, 1, 2, 3],
        chunksize=20_000,
    )

    rows_processed = 0
    stats = []
    for chunk in chunks:
        stats_chunk = []
        for row in chunk.itertuples(index=False):
            totals, goals = _extract_total_goal(row.totals_json)
            total_cal = totals.get("calories")
            total_carb = totals.get("carbs")
            total_sugar = totals.get("sugar")
            total_protein = totals.get("protein")
            goal_cal = goals.get("calories")
            if total_cal is None:
                continue
            stats_chunk.append(
                {
                    "user_id": int(row.user_id),
                    "calories": float(total_cal),
                    "carbs": float(total_carb or 0),
                    "sugar": float(total_sugar or 0),
                    "protein": float(total_protein or 0),
                    "goal_calories": float(goal_cal) if goal_cal is not None else np.nan,
                }
            )
        if stats_chunk:
            stats.extend(stats_chunk)
        rows_processed += len(chunk)
        if rows_processed >= max_rows:
            break

    stat_df = pd.DataFrame(stats)
    if stat_df.empty:
        raise ValueError("MyFitnessPal TSV was read but produced no usable nutrition totals.")

    aggregated = (
        stat_df.groupby("user_id")
        .agg(
            calories=("calories", "mean"),
            carbs=("carbs", "mean"),
            sugar=("sugar", "mean"),
            protein=("protein", "mean"),
            goal_calories=("goal_calories", "mean"),
            days_logged=("calories", "count"),
        )
        .reset_index()
    )
    aggregated = aggregated[aggregated["days_logged"] >= 3].copy()
    aggregated["sugar_ratio"] = aggregated["sugar"] / (aggregated["carbs"] + 1e-6)
    return aggregated


def build_diet_profiles(nhanes_profiles: pd.DataFrame, mfp_agg: pd.DataFrame) -> pd.DataFrame:
    n = min(len(nhanes_profiles), len(mfp_agg))
    if n < 100:
        raise ValueError("Not enough merged records to build diet profiles.")

    nh = nhanes_profiles.sample(n=n, random_state=42).reset_index(drop=True)
    mfp = mfp_agg.sample(n=n, random_state=42).reset_index(drop=True)

    q_sugar = mfp["sugar_ratio"].quantile(0.65)
    q_protein = mfp["protein"].quantile(0.65)
    q_days = mfp["days_logged"].quantile(0.40)
    q_cal = mfp["calories"].quantile(0.55)
    q_goal_gap = (mfp["goal_calories"] - mfp["calories"]).fillna(0)

    out = pd.DataFrame(
        {
            "profile_id": [f"user_{uid}" for uid in mfp["user_id"].astype(int)],
            "age": nh["age"].astype(int),
            "height_cm": nh["height_cm"].astype(float).round(1),
            "weight_kg": nh["weight_kg"].astype(float).round(1),
            "sex_bin": nh["sex_bin"].astype(int),
            "night_shift": (mfp["days_logged"] < q_days).astype(int),
            "sugar_craving": (mfp["sugar_ratio"] > q_sugar).astype(int),
            "home_workout": ((mfp["days_logged"] % 2) == 0).astype(int),
            "vegetarian_pref": ((mfp["protein"] < q_protein) & (mfp["sugar_ratio"] < q_sugar)).astype(int),
            "high_stress": (mfp["calories"] > q_cal).astype(int),
            "short_sessions": ((mfp["days_logged"] % 3) == 0).astype(int),
            "goal_direction": np.sign(q_goal_gap).astype(int),
            "adherence_score": np.clip(
                100 - (np.abs(q_goal_gap) / (mfp["goal_calories"].fillna(mfp["calories"]) + 1e-6) * 100),
                45,
                99,
            ).round(1),
        }
    )

    macro_balance = mfp["protein"] / (mfp["calories"] + 1e-6)
    out["diet_pattern"] = np.where(
        out["sugar_craving"] == 1,
        "high_sugar_snacker",
        np.where(macro_balance > macro_balance.median(), "higher_protein", "mixed_balanced"),
    )
    out["notes"] = (
        "Built from NHANES body stats + MyFitnessPal diary aggregates; "
        "lifestyle flags inferred from logging and macro patterns."
    )
    return out[
        [
            "profile_id",
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
            "diet_pattern",
            "adherence_score",
            "notes",
        ]
    ]


def build_megagym_subset(mega_path: Path) -> pd.DataFrame:
    gym = pd.read_csv(mega_path)
    level_map = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
    duration_map = {"Strength": 40, "Cardio": 30, "Stretching": 20, "Plyometrics": 25, "Strongman": 45}

    out = pd.DataFrame(
        {
            "exercise_name": gym["Title"].astype(str),
            "muscle_group": gym["BodyPart"].fillna("full body").astype(str).str.lower(),
            "difficulty": gym["Level"].map(level_map).fillna(2).astype(int),
            "equipment": gym["Equipment"]
            .fillna("bodyweight")
            .astype(str)
            .str.lower()
            .replace({"body only": "bodyweight", "none": "bodyweight"}),
            "duration_min": gym["Type"].map(duration_map).fillna(30).astype(int),
        }
    )
    out = out.drop_duplicates(subset=["exercise_name"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full datasets for Diet Twin app.")
    parser.add_argument("--demo-xpt", type=Path, required=True)
    parser.add_argument("--bmx-xpt", type=Path, required=True)
    parser.add_argument("--mfp-tsv", type=Path, required=True)
    parser.add_argument("--megagym-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--mfp-max-rows", type=int, default=400_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nhanes = load_nhanes_profiles(args.demo_xpt, args.bmx_xpt)
    mfp_agg = load_mfp_aggregates(args.mfp_tsv, max_rows=args.mfp_max_rows)
    diet_profiles = build_diet_profiles(nhanes, mfp_agg)
    megagym = build_megagym_subset(args.megagym_csv)

    nhanes.to_csv(args.output_dir / "nhanes_body_profiles.csv", index=False)
    diet_profiles.to_csv(args.output_dir / "diet_lifestyle_profiles.csv", index=False)
    megagym.to_csv(args.output_dir / "megagym_subset.csv", index=False)

    print("Wrote datasets:")
    print(" -", args.output_dir / "nhanes_body_profiles.csv", f"({len(nhanes)} rows)")
    print(" -", args.output_dir / "diet_lifestyle_profiles.csv", f"({len(diet_profiles)} rows)")
    print(" -", args.output_dir / "megagym_subset.csv", f"({len(megagym)} rows)")


if __name__ == "__main__":
    main()
