import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients")

all_modes = {}

for f in ROOT.glob("*_client.csv"):
    df = pd.read_csv(f, usecols=["city", "mode"], dtype=str, low_memory=False)
    city = f.stem.replace("_client", "")
    counts = df["mode"].value_counts(dropna=False)
    all_modes[city] = set(df["mode"].dropna().unique())

    print("\n" + "=" * 60)
    print(city.upper())
    print(counts)

common = set.intersection(*all_modes.values()) if all_modes else set()
union = set.union(*all_modes.values()) if all_modes else set()

print("\n" + "=" * 60)
print("COMMON MODES ACROSS ALL CITIES:", sorted(common))
print("ALL MODES SEEN:", sorted(union))