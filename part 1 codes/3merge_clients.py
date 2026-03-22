import pandas as pd
from pathlib import Path
#After splitting the raw data by city, I merged the multiple files within each city into one client dataset.
ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city")

OUTPUT = ROOT / "clients"
OUTPUT.mkdir(exist_ok=True)

for city_folder in ROOT.iterdir():

    if not city_folder.is_dir():
        continue

    city = city_folder.name
    csv_files = list(city_folder.glob("*.csv"))

    if len(csv_files) == 0:
        continue

    print(f"\nProcessing {city} ({len(csv_files)} files)")

    dfs = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["city"] = city
            dfs.append(df)
        except:
            print("Skipped:", f.name)

    merged = pd.concat(dfs, ignore_index=True)

    save_path = OUTPUT / f"{city}_client.csv"
    merged.to_csv(save_path, index=False)

    print("Saved:", save_path)