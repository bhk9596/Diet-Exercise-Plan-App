import pandas as pd
import json
from pathlib import Path
import os

def parse_mfp_summary(summary_str):
    """
    Parses the JSON string in the summary column and extracts Calories, Protein, Carbs, and Fat.
    """
    try:
        data = json.loads(summary_str)
        # The JSON looks like: {"total": [{"name": "Calories", "value": 2001}, ...]}
        totals = data.get("total", [])
        
        extracted = {"Calories": 0, "Protein": 0, "Carbs": 0, "Fat": 0}
        for item in totals:
            name = item.get("name")
            value = item.get("value", 0)
            if name in extracted:
                extracted[name] = value
        return pd.Series(extracted)
    except Exception:
        # If parsing fails or data is missing, return None
        return pd.Series({"Calories": None, "Protein": None, "Carbs": None, "Fat": None})


def clean_mfp_data(input_path, output_path):
    print(f"Reading first 50,000 rows from {input_path}...")
    # The MFP dataset doesn't have headers based on initial inspection.
    df_mfp = pd.read_csv(
        input_path, 
        sep='\t', 
        nrows=50000, 
        header=None,
        names=['user_id', 'date', 'meals', 'summary'],
        on_bad_lines='skip'
    )
    
    print("Parsing JSON summaries for macronutrients...")
    # Apply the JSON parser to the summary column
    macros_df = df_mfp['summary'].apply(parse_mfp_summary)
    
    # Combine the parsed macros with the user_id
    combined_df = pd.concat([df_mfp[['user_id']], macros_df], axis=1)
    
    # Drop rows where parsing failed
    combined_df = combined_df.dropna()
    
    print("Grouping by user_id and calculating means...")
    # Group by user_id to get stable daily averages
    user_profiles = combined_df.groupby('user_id').mean().reset_index()
    
    # Round the values to 2 decimal places for cleaner data
    user_profiles = user_profiles.round(2)
    
    print(f"Saving MFP profiles to {output_path}...")
    user_profiles.to_csv(output_path, index=False)
    print(f"Done! Saved {len(user_profiles)} unique user profiles.\n")


def clean_food_data(input_path, output_path):
    print(f"Reading Food.com dataset from {input_path}...")
    # Select only the relevant columns to save memory
    columns_to_keep = [
        'RecipeId', 'Name', 'Calories', 'FatContent', 
        'ProteinContent', 'CarbohydrateContent'
    ]
    
    df_food = pd.read_csv(input_path, usecols=columns_to_keep)
    
    # Rename columns to standard names
    rename_mapping = {
        'FatContent': 'Fat',
        'ProteinContent': 'Protein',
        'CarbohydrateContent': 'Carbs'
    }
    df_food = df_food.rename(columns=rename_mapping)
    
    print("Filtering outliers (>3000 Calories) and dropping missing values...")
    # Filter out extreme outliers and NaNs
    df_food = df_food.dropna(subset=['Calories', 'Fat', 'Protein', 'Carbs'])
    df_food = df_food[df_food['Calories'] <= 3000]
    
    print("Sampling 10,000 random recipes...")
    # Ensure we don't try to sample more than what's available
    n_sample = min(10000, len(df_food))
    df_sample = df_food.sample(n=n_sample, random_state=42)
    
    print(f"Saving Food catalog to {output_path}...")
    df_sample.to_csv(output_path, index=False)
    print("Done!\n")


if __name__ == "__main__":
    # Define file paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    # Task 1 Paths
    mfp_input = data_dir / "mfp-diaries.tsv"
    mfp_output = data_dir / "clean_mfp_profiles.csv"
    
    # Task 2 Paths
    food_input = data_dir / "The Food.com.csv"
    food_output = data_dir / "clean_food_catalog.csv"
    
    # Execute Task 1
    if mfp_input.exists():
        clean_mfp_data(mfp_input, mfp_output)
    else:
        print(f"Error: Could not find {mfp_input}")
        
    # Execute Task 2
    if food_input.exists():
        clean_food_data(food_input, food_output)
    else:
        print(f"Error: Could not find {food_input}")
