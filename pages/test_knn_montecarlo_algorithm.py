import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import diet_twin_finder
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from diet_twin_finder import DietTwinFinder
from meal_generator import MealGenerator
import numpy as np
from app import estimate_calories, get_activity_multiplier

DIET_FEATURE_COLS = [
    'age', 'height_cm', 'weight_kg', 'sex_bin',
    'night_shift', 'diet_pattern_enc', 'home_workout',
    'vegetarian_pref', 'high_stress', 'short_sessions', 'goal_direction',
    'adherence_score'
]
DIET_FEATURE_WEIGHTS = np.array([
    1.0, 1.0, 1.0, 1.0,    # body profile base weights
    1.2, 1.4, 1.0,         # lifestyle: night_shift, diet_pattern_enc (3-level richer signal), home_workout
    1.0, 1.5, 1.0,         # behavioral flags (stress is highly weighted)
    1.5,                   # goal direction is crucial
    1.5                    # adherence capacity: matching realistic behavioral ceiling is equally critical
])
st.set_page_config(page_title="Algorithm Testing Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Diet Recommendation Algorithm Testing")
st.markdown("This dashboard perfectly reflects the 3-input architecture defined in our project methods.")

@st.cache_data
def load_profiles():
    data_path = parent_dir / "data" / "diet_lifestyle_profiles.csv"
    return pd.read_csv(data_path)

@st.cache_data
def load_food_catalog():
    data_path = parent_dir / "data" / "clean_food_catalog.csv"
    return pd.read_csv(data_path)

@st.cache_data
def load_activity_multipliers():
    data_path = parent_dir / "data" / "activity_multipliers.csv"
    return pd.read_csv(data_path)

try:
    df = load_profiles()
    food_df = load_food_catalog()
    activity_df = load_activity_multipliers()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Encode diet_pattern as ordinal: higher_protein=0, mixed_balanced=1, high_sugar_snacker=2
DIET_PATTERN_MAP = {'higher_protein': 0, 'mixed_balanced': 1, 'high_sugar_snacker': 2}
df['diet_pattern_enc'] = df['diet_pattern'].map(DIET_PATTERN_MAP).fillna(1)

# Extract numerical columns to ensure we build the user_vector in the exact correct order
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'profile_id' in numerical_cols:
    numerical_cols.remove('profile_id')
if 'id' in numerical_cols:
    numerical_cols.remove('id')

# Sidebar: Hyperparameters
st.sidebar.header("Hyperparameters")
metric = st.sidebar.selectbox("Distance Metric", ["cosine", "euclidean", "manhattan"])
k = st.sidebar.slider("Number of Twins (k)", min_value=1, max_value=10, value=1)

st.markdown("---")

# ==========================================
# INPUT 1: Physical Measurements
# ==========================================
st.header("Input 1: Physical Measurements (Simulated)")
col1_1, col1_2, col1_3, col1_4 = st.columns(4)

age_min, age_max, age_mean = float(df['age'].min()), float(df['age'].max()), float(df['age'].mean())
height_min, height_max, height_mean = float(df['height_cm'].min()), float(df['height_cm'].max()), float(df['height_cm'].mean())
weight_min, weight_max, weight_mean = float(df['weight_kg'].min()), float(df['weight_kg'].max()), float(df['weight_kg'].mean())

age = col1_1.slider("Age", min_value=age_min, max_value=age_max, value=age_mean, step=1.0)
height_cm = col1_2.slider("Height (cm)", min_value=height_min, max_value=height_max, value=height_mean, step=0.1)
weight_kg = col1_3.slider("Weight (kg)", min_value=weight_min, max_value=weight_max, value=weight_mean, step=0.1)
sex = col1_4.selectbox("Biological Sex", ["Male", "Female"])

st.markdown("---")

# ==========================================
# INPUT 2: Lifestyle Data
# ==========================================
st.header("Input 2: Lifestyle Data")

col2_1, col2_2, col2_3 = st.columns(3)
night_shift = col2_1.radio("Works Night Shifts", ["No", "Yes"])
diet_pattern_sel = col2_2.radio(
    "Diet Pattern",
    ["💪 Higher Protein", "🥗 Mixed / Balanced", "🍬 High Sugar Snacker"],
    help="Replaces simple sugar craving flag — captures eating style at 3 levels"
)
high_stress = col2_3.radio("High Stress Level", ["No", "Yes"])

col2_4, col2_5, col2_6 = st.columns(3)
vegetarian_pref = col2_4.radio("Vegetarian Preference", ["No", "Yes"])
workout_pref = col2_5.radio("Workout Preference", ["Gym", "Home Workout"])
session_length = col2_6.radio("Session Length", ["Standard", "Short (<30 mins)"])

st.markdown("---")

# ==========================================
# INPUT 3: Goals & Expectations
# ==========================================
st.header("Input 3: Goals & Expectations")
col3_1, col3_2 = st.columns(2)

goal = col3_1.selectbox("Goal Direction", ["Maintenance", "Weight Loss", "Muscle Gain"])

adh_min, adh_max, adh_mean = float(df['adherence_score'].min()), float(df['adherence_score'].max()), float(df['adherence_score'].mean())
adherence_score = col3_2.slider("Diet Adherence Score (Self-assessed)", min_value=adh_min, max_value=adh_max, value=adh_mean, step=1.0)

# Build the feature dictionary mapping
_diet_pattern_enc_val = {
    "💪 Higher Protein": 0.0,
    "🥗 Mixed / Balanced": 1.0,
    "🍬 High Sugar Snacker": 2.0,
}[diet_pattern_sel]

user_input_map = {
    'age': age,
    'height_cm': height_cm,
    'weight_kg': weight_kg,
    'sex_bin': 1.0 if sex == "Female" else 0.0,
    'night_shift': 1.0 if night_shift == "Yes" else 0.0,
    'diet_pattern_enc': _diet_pattern_enc_val,
    'high_stress': 1.0 if high_stress == "Yes" else 0.0,
    'vegetarian_pref': 1.0 if vegetarian_pref == "Yes" else 0.0,
    'home_workout': 1.0 if workout_pref == "Home Workout" else 0.0,
    'short_sessions': 1.0 if session_length == "Short (<30 mins)" else 0.0,
    'goal_direction': 0.0 if goal == "Maintenance" else (-1.0 if goal == "Weight Loss" else 1.0),
    'adherence_score': adherence_score
}

# Construct the user vector ensuring strict column order matches DIET_FEATURE_COLS
user_vector = [user_input_map[col] for col in DIET_FEATURE_COLS]

st.markdown("---")

# ==========================================
# OUTPUT: Twin Retrieval Results (k-NN)
# ==========================================
st.header("Twin Retrieval Results (Algorithm Output)")

with st.spinner("Finding twins..."):
    # Initialize the finder with the selected metric
    finder = DietTwinFinder(df, metric=metric, feature_cols=DIET_FEATURE_COLS, weights=DIET_FEATURE_WEIGHTS)
    
    # Find the nearest twin(s)
    indices, distances = finder.find_twin(user_vector, k=k)

# Display Results
for i, (idx, dist) in enumerate(zip(indices, distances)):
    twin_row = df.iloc[idx]
    
    st.markdown(f"### Twin #{i+1} ({twin_row.get('profile_id', 'Unknown')})")
    
    # Convert Cosine Distance to Cosine Similarity (1 - distance)
    if metric == "cosine":
        similarity = 1.0 - dist
        st.metric(label="Cosine Similarity Match", value=f"{similarity:.4f}")
    else:
        st.metric(label=f"Cost / Loss ({metric} distance)", value=f"{dist:.4f}")
        
    # --- Calculate Macro Targets Scientifically ---
    try:
        # Determine goal weight based on selected goal
        # (Use twin's weight as the base to calculate their goal weight)
        twin_weight = float(twin_row.get("weight_kg", weight_kg))
        twin_height = float(twin_row.get("height_cm", height_cm))
        twin_age = int(twin_row.get("age", age))
        twin_sex_bin = int(twin_row.get("sex_bin", 0))
        twin_sex_str = "F" if twin_sex_bin == 1 else "M"

        if goal == "Weight Loss":
            goal_weight_kg = weight_kg - 5.0
        elif goal == "Muscle Gain":
            goal_weight_kg = weight_kg + 5.0
        else:
            goal_weight_kg = weight_kg
            
        # Default days_per_week for the testing dashboard
        days_per_week = 4
        
        # Calculate using the USER's actual body (matches app.py)
        sex_str = "M" if sex == "Male" else "F"
        calc_cals = estimate_calories(age, sex_str, height_cm, weight_kg, goal_weight_kg, days_per_week, activity_df)
        
        # Determine AMDR Hinge Loss Ranges based on Diet Pattern Selection
        # Base AMDR Ranges
        pro_pct = [0.20, 0.35]
        carb_pct = [0.40, 0.55]
        fat_pct = [0.25, 0.35]

        if diet_pattern_sel == "💪 Higher Protein":
            pro_pct = [0.36, 0.50]
            carb_pct = [0.30, 0.44]
            fat_pct = [0.20, 0.30]
        # In the test dashboard, "Low Carb" isn't a direct radio option, it falls under specific UI tests.
        # But for mathematical parity, we calculate the absolute grams:
        
        calc_pro_min, calc_pro_max = round(calc_cals * pro_pct[0] / 4), round(calc_cals * pro_pct[1] / 4)
        calc_carbs_min, calc_carbs_max = round(calc_cals * carb_pct[0] / 4), round(calc_cals * carb_pct[1] / 4)
        calc_fat_min, calc_fat_max = round(calc_cals * fat_pct[0] / 9), round(calc_cals * fat_pct[1] / 9)

        st.markdown("#### Scientifically Calculated AMDR Macro Ranges")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Calories", f"{calc_cals} kcal")
        m_col2.metric("Protein Range", f"{calc_pro_min}-{calc_pro_max} g")
        m_col3.metric("Carbs Range", f"{calc_carbs_min}-{calc_carbs_max} g")
        m_col4.metric("Fat Range", f"{calc_fat_min}-{calc_fat_max} g")
        
        # Save the exact AMDR boundaries to session state for the sliders
        st.session_state["calc_pro_range"] = (int(calc_pro_min), int(calc_pro_max))
        st.session_state["calc_carbs_range"] = (int(calc_carbs_min), int(calc_carbs_max))
        st.session_state["calc_fat_range"] = (int(calc_fat_min), int(calc_fat_max))
        
    except Exception as e:
        st.warning(f"Could not calculate macros: {e}")
    # -------------------------------------------------------------------------
        
    # Create a comparison dataframe for visualization
    comparison_df = pd.DataFrame({
        'Feature': DIET_FEATURE_COLS,
        'User Input': user_vector,
        'Twin Data': [twin_row[col] for col in DIET_FEATURE_COLS]
    }).set_index('Feature')
    
    # Show comparison chart
    st.bar_chart(comparison_df)
    
    # Show raw twin data for reference
    with st.expander("View Raw Twin Profile"):
        st.dataframe(twin_row.to_frame().T)

st.markdown("---")

# ==========================================
# OUTPUT: Meal Generator (Monte Carlo)
# ==========================================
st.header("Meal Plan Generator (Monte Carlo Search)")
st.markdown("Input target macros. **Total Calories are automatically calculated** to ensure mathematical consistency.")

# Use calculated AMDR ranges if available
default_pro_range = st.session_state.get("calc_pro_range", (110, 160))
default_carbs_range = st.session_state.get("calc_carbs_range", (200, 250))
default_fat_range = st.session_state.get("calc_fat_range", (60, 90))

# Macro Range Input Sliders (AMDR Hinge Loss Optimization)
mcol1, mcol2, mcol3 = st.columns(3)

pro_range = mcol1.slider("Protein Range (g)", 30, 400, default_pro_range, step=5)
carb_range = mcol2.slider("Carbs Range (g)", 50, 600, default_carbs_range, step=10)
fat_range = mcol3.slider("Fat Range (g)", 20, 300, default_fat_range, step=5)

# Mathematically valid calorie calculation using the midpoints of the selected ranges
mid_pro = (pro_range[0] + pro_range[1]) / 2.0
mid_carbs = (carb_range[0] + carb_range[1]) / 2.0
mid_fat = (fat_range[0] + fat_range[1]) / 2.0
calculated_cals = (mid_pro * 4) + (mid_carbs * 4) + (mid_fat * 9)
st.info(f"💡 **Estimated Target Calories:** {calculated_cals:.0f} kcal (Calculated from range midpoints)")

if st.button("Generate Meal Plan"):
    with st.spinner("Running deep Monte Carlo simulation (10000 iterations)..."):
        # Apply Vegetarian Filter (Syncs with app.py logic)
        meal_food_df = food_df.copy()
        if vegetarian_pref == "Yes" and "vegetarian" in meal_food_df.columns:
            veg_only = meal_food_df[meal_food_df["vegetarian"] == 1]
            if not veg_only.empty:
                meal_food_df = veg_only

        generator = MealGenerator(meal_food_df)
        best_plan_df, best_error, actual_totals, error_history = generator.generate_meal_plan(
            calculated_cals, pro_range, carb_range, fat_range, num_meals=7, iterations=10000
        )
        
        st.success(f"Meal Plan Generated! Total Error Score: {best_error:.4f}")
        
        # Display Monte Carlo Convergence Chart
        st.write("### Optimization Convergence (10,000 Iterations)")
        # Cap max value at 1.0 to prevent the +1000.0 absolute penalty spikes from squashing the chart scale
        capped_history = [min(e, 1.0) for e in error_history]
        st.line_chart(capped_history)
        
        # Display Totals Comparison
        st.write("### Target vs. Actual Macros")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        
        # Removed delta arguments to avoid Streamlit UI confusion
        tcol1.metric("Calories", f"{actual_totals['Calories']:.0f} kcal")
        tcol2.metric("Protein", f"{actual_totals['Protein']:.0f} g")
        tcol3.metric("Carbs", f"{actual_totals['Carbs']:.0f} g")
        tcol4.metric("Fat", f"{actual_totals['Fat']:.0f} g")
        
        # Display Recommended Meals — aggregated into 3 meals
        st.write("### Recommended Daily Meal Plan")
        
        dishes = list(best_plan_df.itertuples(index=False))
        meal_groups = {
            "🌅 Breakfast": dishes[0:2],
            "☀️ Lunch":     dishes[2:5],
            "🌙 Dinner":    dishes[5:7],
        }
        
        for meal_name, meal_dishes in meal_groups.items():
            total_cal = sum(d.Calories for d in meal_dishes)
            total_pro = sum(d.Protein for d in meal_dishes)
            total_carbs = sum(d.Carbs for d in meal_dishes)
            total_fat = sum(d.Fat for d in meal_dishes)
            dish_names = " + ".join(d.Name for d in meal_dishes)
            
            col_a, col_b = st.columns([2, 1])
            col_a.markdown(f"**{meal_name}**  \n{dish_names}")
            col_b.metric("Total Calories", f"{total_cal:.0f} kcal")
            st.caption(f"Protein: {total_pro:.0f}g | Carbs: {total_carbs:.0f}g | Fat: {total_fat:.0f}g")
            st.divider()
