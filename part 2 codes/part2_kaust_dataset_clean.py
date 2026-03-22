import os
import pandas as pd
import numpy as np

data_path = r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\kaust"
save_path = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local"

os.makedirs(save_path, exist_ok=True)

mode_files = {
    "bike": "bike.csv",
    "bus_colored": "bus_colored_kaust.csv",
    "bus_ondemand": "bus_ondemand.csv",
    "car": "car_kaust.csv",
    "jog": "jog.csv",
    "motorcycle": "motorcycle.csv",
    "scooter": "scooter.csv",
    "walk": "walk_kaust.csv"
}

all_dfs = []
column_report = []

for label, file in mode_files.items():
    file_path = os.path.join(data_path, file)
    print(f"Loading {label} from {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    original_cols = df.shape[1]

    # remove fully empty columns
    df = df.dropna(axis=1, how="all")

    # convert every column to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove columns that became fully NaN after coercion
    df = df.dropna(axis=1, how="all")

    cleaned_cols = df.shape[1]

    # add label
    df["label"] = label

    all_dfs.append(df)

    column_report.append({
        "label": label,
        "original_columns": original_cols,
        "cleaned_columns": cleaned_cols,
        "rows": len(df)
    })

dataset = pd.concat(all_dfs, ignore_index=True)

# remove rows where all feature columns are NaN
feature_cols = [c for c in dataset.columns if c != "label"]
dataset = dataset.dropna(subset=feature_cols, how="all")

# save reports
summary = dataset["label"].value_counts().rename_axis("label").reset_index(name="count")
summary.to_csv(os.path.join(save_path, "part2_data_summary.csv"), index=False)

pd.DataFrame(column_report).to_csv(
    os.path.join(save_path, "part2_column_cleaning_report.csv"), index=False
)

dataset.to_csv(os.path.join(save_path, "kaust_multiclass_dataset_clean.csv"), index=False)

print("\nSaved:")
print("- part2_data_summary.csv")
print("- part2_column_cleaning_report.csv")
print("- kaust_multiclass_dataset_clean.csv")
print("\nFinal shape:", dataset.shape)