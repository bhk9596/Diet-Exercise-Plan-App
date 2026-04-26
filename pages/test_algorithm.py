import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import diet_twin_finder
# and access the data folder properly
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from diet_twin_finder import DietTwinFinder

st.set_page_config(page_title="Algorithm Testing Dashboard", layout="wide")

st.title("Diet Recommendation Algorithm Testing")
st.markdown("Use this dashboard to tune hyperparameters and visualize the k-NN distance (Cost/Loss).")

@st.cache_data
def load_profiles():
    data_path = parent_dir / "data" / "diet_lifestyle_profiles.csv"
    return pd.read_csv(data_path)

try:
    df = load_profiles()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Extract numerical columns to know what sliders to build
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'profile_id' in numerical_cols:
    numerical_cols.remove('profile_id')
if 'id' in numerical_cols:
    numerical_cols.remove('id')

# Sidebar: Hyperparameters
st.sidebar.header("Hyperparameters")
metric = st.sidebar.selectbox("Distance Metric", ["cosine", "euclidean", "manhattan"])
k = st.sidebar.slider("Number of Twins (k)", min_value=1, max_value=10, value=1)

# Main Area: Dummy User Input
st.subheader("Simulate User Input")
st.markdown("Adjust these values to represent a new user. The algorithm will find the closest twin.")

col1, col2, col3 = st.columns(3)
user_input = {}

# Dynamically generate sliders for each numerical feature based on dataset min/max
cols = [col1, col2, col3]
for i, col_name in enumerate(numerical_cols):
    min_val = float(df[col_name].min())
    max_val = float(df[col_name].max())
    mean_val = float(df[col_name].mean())
    
    # Handle edge case where min == max
    if min_val == max_val:
        max_val = min_val + 1.0
        
    with cols[i % 3]:
        # Use appropriate step size
        step = 1.0 if pd.api.types.is_integer_dtype(df[col_name]) else (max_val - min_val) / 100.0
        user_input[col_name] = st.slider(
            col_name, 
            min_value=min_val, 
            max_value=max_val, 
            value=mean_val,
            step=step
        )

# Construct user vector in the correct order
user_vector = [user_input[col] for col in numerical_cols]

st.markdown("---")

# Run Algorithm
st.subheader("Algorithm Results")

with st.spinner("Finding twins..."):
    # Initialize the finder with the selected metric
    finder = DietTwinFinder(df, metric=metric)
    
    # Find the nearest twin(s)
    indices, distances = finder.find_twin(user_vector, k=k)

# Display Results
for i, (idx, dist) in enumerate(zip(indices, distances)):
    twin_row = df.iloc[idx]
    
    st.markdown(f"### Twin #{i+1}")
    st.metric(label="Cost / Loss (Distance)", value=f"{dist:.4f}")
    
    # Create a comparison dataframe for visualization
    comparison_df = pd.DataFrame({
        'Feature': numerical_cols,
        'User Input': user_vector,
        'Twin Data': [twin_row[col] for col in numerical_cols]
    }).set_index('Feature')
    
    # Show comparison chart
    st.bar_chart(comparison_df)
    
    # Show raw twin data for reference
    with st.expander("View Raw Twin Profile"):
        st.dataframe(twin_row.to_frame().T)
