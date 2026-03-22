import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients")
OUT = ROOT / "fl_ready"
OUT.mkdir(exist_ok=True)

CORE_MODES = {"car", "walk"}
MODE_MAP = {
    "bus_colored": "bus",
    "bus_ondemand": "bus",
}

def normalize_mode(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    return MODE_MAP.get(x, x)

def time_to_seconds(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, format="%H:%M:%S.%f", errors="coerce")
    return (
        dt.dt.hour * 3600
        + dt.dt.minute * 60
        + dt.dt.second
        + dt.dt.microsecond / 1e6
    )

print("STARTED FULL PREP")
print("ROOT exists:", ROOT.exists())
print("ROOT path:", ROOT)

files = sorted(list(ROOT.glob("*_client.csv")))
print("Found client files:", len(files))
for f in files:
    print("-", f.name)

if len(files) == 0:
    raise FileNotFoundError("No *_client.csv files found in the clients folder.")

client_frames = {}
client_columns = []

for f in files:
    print(f"\nLoading {f.name} ...")
    df = pd.read_csv(f, dtype=str, low_memory=False)

    if "mode" not in df.columns or "city" not in df.columns:
        print(f"Skipped {f.name}: missing mode/city")
        continue

    df["mode"] = df["mode"].apply(normalize_mode)
    df = df[df["mode"].isin(CORE_MODES)].copy()

    city = f.stem.replace("_client", "")
    client_frames[city] = df
    client_columns.append(set(df.columns))

    print(f"{city}: rows after keeping only car/walk = {len(df)}")

if not client_frames:
    raise ValueError("No usable client data after filtering to car/walk.")

common_cols = set.intersection(*client_columns)
feature_cols = sorted(list(common_cols - {"city", "mode", "source_file"}))

add_time_seconds = False
if "Time" in feature_cols:
    feature_cols.remove("Time")
    add_time_seconds = True

print("\nCommon columns across all clients:", len(common_cols))
print("Initial feature columns:", len(feature_cols))
print("Will add Time_seconds:", add_time_seconds)

combined_parts = []

for city, df in client_frames.items():
    work = df.copy()

    if add_time_seconds:
        work["Time_seconds"] = time_to_seconds(work["Time"])
        use_cols = feature_cols + ["Time_seconds", "mode", "city"]
    else:
        use_cols = feature_cols + ["mode", "city"]

    work = work[use_cols].copy()

    numeric_cols = [c for c in work.columns if c not in ["mode", "city"]]
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    combined_parts.append(work)

combined_df = pd.concat(combined_parts, ignore_index=True)
print("Combined dataframe shape:", combined_df.shape)

feature_only = [c for c in combined_df.columns if c not in ["mode", "city"]]
missing_ratio = combined_df[feature_only].isna().mean()
kept_features = missing_ratio[missing_ratio <= 0.60].index.tolist()

print("Features before missing filter:", len(feature_only))
print("Features kept after >60% missing filter:", len(kept_features))

if not kept_features:
    raise ValueError("No features remained after >60% missing filter.")

combined_df[kept_features] = combined_df[kept_features].fillna(0)

labels = sorted(combined_df["mode"].dropna().unique().tolist())
label_map = {label: idx for idx, label in enumerate(labels)}

with open(OUT / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)

with open(OUT / "feature_cols.json", "w", encoding="utf-8") as f:
    json.dump(kept_features, f, indent=2)

print("Labels:", label_map)

train_parts = {}
test_parts = {}

for city, df in client_frames.items():
    work = df.copy()

    if add_time_seconds:
        work["Time_seconds"] = time_to_seconds(work["Time"])

    use_cols = kept_features + ["mode", "city"]
    work = work[use_cols].copy()

    for c in kept_features:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work[kept_features] = work[kept_features].fillna(0)
    work = work.dropna(subset=["mode"]).copy()
    work["y"] = work["mode"].map(label_map)

    stratify_y = work["y"] if work["y"].nunique() > 1 else None

    train_df, test_df = train_test_split(
        work,
        test_size=0.20,
        random_state=42,
        stratify=stratify_y
    )

    train_parts[city] = train_df
    test_parts[city] = test_df

all_train = pd.concat(train_parts.values(), ignore_index=True)

scaler = MinMaxScaler()
scaler.fit(all_train[kept_features])

for city in train_parts:
    train_df = train_parts[city].copy()
    test_df = test_parts[city].copy()

    train_df[kept_features] = scaler.transform(train_df[kept_features])
    test_df[kept_features] = scaler.transform(test_df[kept_features])

    train_df.to_csv(OUT / f"{city}_train.csv", index=False)
    test_df.to_csv(OUT / f"{city}_test.csv", index=False)

    print(f"\n{city.upper()}")
    print(f"train shape: {train_df.shape}")
    print(f"test shape:  {test_df.shape}")
    print("train label counts:")
    print(train_df["mode"].value_counts())
    print("test label counts:")
    print(test_df["mode"].value_counts())

print("\nSaved FL-ready files to:")
print(OUT)
print("\nNumber of kept features:", len(kept_features))
print("Labels:", label_map)
print("DONE")