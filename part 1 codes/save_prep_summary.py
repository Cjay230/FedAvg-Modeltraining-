import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients\fl_ready")
RESULTS = Path(r"C:\Users\user\Desktop\VIP\federated\results")
RESULTS.mkdir(parents=True, exist_ok=True)

with open(ROOT / "label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

with open(ROOT / "feature_cols.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

cities = ["jeddah", "kaust", "kz", "mekkah"]
rows = []

for city in cities:
    train = pd.read_csv(ROOT / f"{city}_train.csv")
    test = pd.read_csv(ROOT / f"{city}_test.csv")

    train_counts = train["mode"].value_counts().to_dict()
    test_counts = test["mode"].value_counts().to_dict()

    rows.append({
        "city": city,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_car": int(train_counts.get("car", 0)),
        "train_walk": int(train_counts.get("walk", 0)),
        "test_car": int(test_counts.get("car", 0)),
        "test_walk": int(test_counts.get("walk", 0)),
        "num_features": int(len(feature_cols)),
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(RESULTS / "data_summary.csv", index=False)

summary_json = {
    "task": "binary federated transport mode detection",
    "clients": cities,
    "label_map": label_map,
    "num_features": len(feature_cols),
    "feature_cols": feature_cols,
    "per_city_summary": rows,
}

with open(RESULTS / "data_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

print("Saved:")
print(RESULTS / "data_summary.csv")
print(RESULTS / "data_summary.json")
print("\nPreview:")
print(summary_df)