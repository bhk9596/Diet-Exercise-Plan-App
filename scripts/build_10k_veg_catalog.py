import pandas as pd
import re
from pathlib import Path
import html
import random

meat_keywords = [
    "chicken", "beef", "pork", "lamb", "fish", "shrimp", "prawn", "crab", "salmon", 
    "tuna", "bacon", "ham", "sausage", "steak", "duck", "turkey", "meatball", 
    "meatloaf", "veal", "prosciutto", "chorizo", "pepperoni", "salami", "ribs", 
    "clam", "oyster", "scallop", "mussel", "cod", "halibut", "trout", "catfish"
]

pattern = re.compile(r'\b(?:' + '|'.join(meat_keywords) + r')\b', re.IGNORECASE)

input_file = Path("data/The Food.com.csv")
output_file = Path("data/clean_food_catalog.csv")

veg_rows = []
meat_rows = []

print("Reading The Food.com.csv...")
for chunk in pd.read_csv(input_file, chunksize=50000, usecols=["RecipeId", "Name", "Calories", "FatContent", "CarbohydrateContent", "ProteinContent"]):
    chunk = chunk.dropna()
    for _, row in chunk.iterrows():
        if len(veg_rows) >= 10000 and len(meat_rows) >= 3000:
            break
            
        name = html.unescape(str(row["Name"]))
        if pattern.search(name):
            if len(meat_rows) < 3000:
                meat_rows.append({
                    "RecipeId": row["RecipeId"],
                    "Name": name,
                    "Calories": row["Calories"],
                    "Fat": row["FatContent"],
                    "Carbs": row["CarbohydrateContent"],
                    "Protein": row["ProteinContent"],
                    "vegetarian": 0
                })
        else:
            if len(veg_rows) < 10000:
                veg_rows.append({
                    "RecipeId": row["RecipeId"],
                    "Name": name,
                    "Calories": row["Calories"],
                    "Fat": row["FatContent"],
                    "Carbs": row["CarbohydrateContent"],
                    "Protein": row["ProteinContent"],
                    "vegetarian": 1
                })
                
    if len(veg_rows) >= 10000 and len(meat_rows) >= 3000:
        break

final_df = pd.DataFrame(veg_rows + meat_rows)
final_df = final_df.sample(frac=1).reset_index(drop=True)
final_df.to_csv(output_file, index=False)

print(f"Dataset rebuilt: {len(final_df)} total recipes.")
print(f"Vegetarian recipes: {len(veg_rows)}")
print(f"Non-Vegetarian recipes: {len(meat_rows)}")
