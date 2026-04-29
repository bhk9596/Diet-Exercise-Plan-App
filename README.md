# Diet Twin Planner (ML Project Proposal Prototype)

This is a creative website app that follows your ML proposal path:

1. Use physical stats to predict a body profile category.
2. Parse free-text lifestyle constraints.
3. Retrieve a nearest "Diet Twin" with cosine-similarity k-NN.
4. Recommend realistic meals + workouts aligned with schedule/equipment.

## Run locally

```bash
cd Diet-Exercise-Plan-App
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
```Windows:
cd ml_diet_twin_app
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```


## Datasets used

- `data/nhanes_body_profiles.csv` (NHANES-style physical stats subset)
- `data/diet_lifestyle_profiles.csv` (MyFitnessPal-style behavior/profile subset)
- `data/megagym_subset.csv` (MegaGym-style exercise metadata subset)
- `data/food_catalog.csv` (**extra dataset**) nutrient-oriented meal options
- `data/activity_multipliers.csv` (**extra dataset**) activity-to-TDEE factors

## Use full datasets (recommended)

If you downloaded the full source files (for example:
`P_DEMO.xpt`, `P_BMX.xpt`, `mfp-diaries.tsv`, `megaGymDataset.csv`),
you can rebuild the app-ready CSVs with:

```bash
python3 scripts/prepare_full_datasets.py \
  --demo-xpt "/path/to/P_DEMO.xpt" \
  --bmx-xpt "/path/to/P_BMX.xpt" \
  --mfp-tsv "/path/to/mfp-diaries.tsv" \
  --megagym-csv "/path/to/megaGymDataset.csv" \
  --output-dir data
```

This overwrites:
- `data/nhanes_body_profiles.csv`
- `data/diet_lifestyle_profiles.csv`
- `data/megagym_subset.csv`

## Methods implemented

- **Classification**: RandomForest body-type model from age, sex, height, weight.
- **Lifestyle NLP extraction**: keyword features from user narrative (night shift, sugar cravings, stress, etc.).
- **Retrieval**: k-NN with cosine similarity to match the nearest diet twin.
- **Plan generation**:
  - calorie target from BMR/TDEE + goal adjustment,
  - meal picks based on calorie fit and protein bias,
  - workout picks constrained by available equipment and session length.

## Notes

- This is a prototype intended for coursework and product exploration.
- You can replace these sample CSVs with full NHANES/Kaggle datasets for stronger realism.
