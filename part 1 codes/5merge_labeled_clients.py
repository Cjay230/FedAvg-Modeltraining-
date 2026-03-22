import pandas as pd
from pathlib import Path
#This was used to finalize the labeled city datasets before federated preparation.
ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled")
OUT = ROOT / "clients"
OUT.mkdir(exist_ok=True)

for city_folder in ROOT.iterdir():
    if not city_folder.is_dir():
        continue
    if city_folder.name == "clients":
        continue

    city = city_folder.name
    csv_files = list(city_folder.glob("*.csv"))

    if not csv_files:
        continue

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype=str, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Skipped {f.name}: {e}")

    if not dfs:
        continue

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    save_path = OUT / f"{city}_client.csv"
    merged.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")
    print(f"Shape: {merged.shape}")

print("\nFinished merging labeled clients.")