import pandas as pd
from pathlib import Path
#At this stage I rebuilt the city client files so that the transport mode labels are clearly attached and usable for training
ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data")
CELLMOB = ROOT / "CellMob"
ADDITIONAL = ROOT / "KAUST_additional_modes"

OUT = ROOT / "dataset_by_city_labeled"
OUT.mkdir(exist_ok=True)

CITY_DIRS = {
    "kaust": OUT / "kaust",
    "jeddah": OUT / "jeddah",
    "mekkah": OUT / "mekkah",
    "kz": OUT / "kz",
}

for d in CITY_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def get_city_and_mode(folder_name: str):
    name = folder_name.lower()

    # Additional KAUST modes
    if name in ["bike", "bus_ondemand", "jog", "motorcycle", "scooter"]:
        return "kaust", name

    parts = name.split("_")

    # Expected forms like walk_mekkah, car_jeddah, bus_colored_kaust
    if "jeddah" in parts:
        city = "jeddah"
    elif "mekkah" in parts:
        city = "mekkah"
    elif "kaust" in parts:
        city = "kaust"
    elif "kz" in parts:
        city = "kz"
    else:
        return None, None

    # mode = everything except city token
    mode_parts = [p for p in parts if p not in ["jeddah", "mekkah", "kaust", "kz"]]
    mode = "_".join(mode_parts)

    return city, mode

def process_folder(folder: Path):
    city, mode = get_city_and_mode(folder.name)
    if city is None or mode is None:
        print(f"Skipped folder: {folder.name}")
        return

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        return

    dfs = []

    for f in csv_files:
        if f.name.startswith("._"):
            continue
        try:
            df = pd.read_csv(f, dtype=str, low_memory=False)
            df["city"] = city
            df["mode"] = mode
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"Skipped file {f.name}: {e}")

    if not dfs:
        return

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    save_path = CITY_DIRS[city] / f"{folder.name}.csv"
    merged.to_csv(save_path, index=False)
    print(f"Saved labeled file: {save_path}")

# Process CellMob folders
for folder in CELLMOB.iterdir():
    if folder.is_dir():
        process_folder(folder)

# Process additional KAUST modes
for folder in ADDITIONAL.iterdir():
    if folder.is_dir():
        process_folder(folder)

print("\nFinished building labeled city folders.")