import os
import pandas as pd
import numpy as np
#this script loads the separate KAUST mode files, assigns the correct label to each one, and merges them into one combined multi-class dataset.
# -------- PATHS --------
data_path = r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\kaust"
save_path = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local"

os.makedirs(save_path, exist_ok=True)

# -------- FILES --------
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

for label, file in mode_files.items():

    file_path = os.path.join(data_path, file)

    print(f"Loading {label}")

    df = pd.read_csv(file_path)

    # Remove fully empty columns (caused by commas issue)
    df = df.dropna(axis=1, how='all')

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Add label column
    df["label"] = label

    all_dfs.append(df)

# -------- MERGE DATA --------
dataset = pd.concat(all_dfs, ignore_index=True)

# -------- DATA SUMMARY --------
summary = dataset["label"].value_counts()

print("\nClass distribution:")
print(summary)

summary.to_csv(os.path.join(save_path, "part2_data_summary.csv"))

dataset.to_csv(os.path.join(save_path, "kaust_multiclass_dataset.csv"), index=False)

print("\nDataset saved.")