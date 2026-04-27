import pandas as pd
import json
from pathlib import Path
import os


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
    
    # Task Paths
    food_input = data_dir / "The Food.com.csv"
    food_output = data_dir / "clean_food_catalog.csv"
        
    # Execute Task 2
    if food_input.exists():
        clean_food_data(food_input, food_output)
    else:
        print(f"Error: Could not find {food_input}")
