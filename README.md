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

## Repo structure
```text
Diet-Exercise-Plan-App/
├── app.py                     # Main Streamlit application and ML orchestration
├── diet_twin_finder.py        # Custom pure-NumPy k-NN implementation for Diet Twin
├── meal_generator.py          # Custom pure-NumPy Monte Carlo optimizer for Meal Planning
├── ui_sections.py             # Streamlit UI components and layout rendering
├── pages/
│   └── test_algorithm.py      # Standalone testing dashboard to validate math/algorithms
├── data/                      # Directory for CSV datasets (NHANES, diets, workouts)
├── scripts/                   # Utility scripts for data cleaning and preparation
└── README.md                  # Project overview and run instructions
```

## Division of Labor

*   **Boren**: Streamlit UI/UX architecture and overall frontend design (`ui_sections.py`).
*   **Jiahao**: Engineered the 3 Input modules, parsing physical stats and lifestyle questionnaire data into machine-readable structures.
*   **Gaohong**: Implemented the Diet output module. Built the `DietTwinFinder` (k-NN retrieval using pure NumPy and Cosine similarity) and the `MealGenerator` (Monte Carlo optimization algorithm).
*   **Jason**: Implemented the Exercise output module. Developed the Safe Workout Picker using rule-based injury filtering.
*   **Ben**: Implemented the Lifestyle Fit output module. Developed the RandomForest models for body-type classification and lifestyle pattern prediction.

## Methods implemented

- **Diet Twin Retrieval (From Scratch )**: k-NN using purely NumPy. Computes Standardisation, applies behavioural weighting, and uses Cosine Similarity to find the most behaviourally similar user.
- **Meal Generator**: Monte Carlo optimization algorithm using pure NumPy/Pandas. Runs 10,000 iterations to minimize macro target errors to find the optimal 7-dish combination.
- **Classification & Regression**: RandomForest model predicting body-type and lifestyle fit scores from user vectors.
- **Safe Workout Picker**: Strict injury rule-based filtering to select home/gym workouts avoiding compromised joints.

## Notes

- This is a prototype intended for coursework and product exploration.
- You can replace these sample CSVs with full NHANES/Kaggle datasets for stronger realism.
