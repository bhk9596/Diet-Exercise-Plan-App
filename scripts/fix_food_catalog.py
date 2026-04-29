import html
import csv
import re
from pathlib import Path

csv_path = Path(__file__).parent.parent / "data" / "clean_food_catalog.csv"

# Read
with open(csv_path, encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Decode HTML entities in every string cell
fixed = 0
for row in rows:
    for key in row:
        original = row[key]
        decoded = html.unescape(original)
        if decoded != original:
            row[key] = decoded
            fixed += 1

print(f"Fixed {fixed} cells across {len(rows)} rows")

# Write back
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Verify
html_entity = re.compile(r"&[A-Za-z]+;|&#\d+;")
with open(csv_path, encoding="utf-8") as f:
    rows2 = list(csv.DictReader(f))

still_bad = [r for r in rows2 if any(html_entity.search(str(v)) for v in r.values())]
print(f"Remaining HTML entity rows after fix: {len(still_bad)}")
print("Sample check row 2:", rows2[2]["Name"])
print("Sample check row 8:", rows2[8]["Name"])
