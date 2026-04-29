import pandas as pd
from pathlib import Path
import re

csv_path = Path("data/clean_food_catalog.csv")
df = pd.read_csv(csv_path)

meat_keywords = [
    "chicken", "beef", "pork", "lamb", "fish", "shrimp", "prawn", "crab", "salmon", 
    "tuna", "bacon", "ham", "sausage", "steak", "duck", "turkey", "meatball", 
    "meatloaf", "veal", "prosciutto", "chorizo", "pepperoni", "salami", "ribs", 
    "clam", "oyster", "scallop", "mussel", "cod", "halibut", "trout", "catfish"
]

pattern = re.compile(r'\b(?:' + '|'.join(meat_keywords) + r')\b', re.IGNORECASE)

df['vegetarian'] = df['Name'].apply(lambda x: 0 if pattern.search(str(x)) else 1)

df.to_csv(csv_path, index=False)
print("Added vegetarian column. Total recipes:", len(df))
print("Vegetarian recipes:", df['vegetarian'].sum())
