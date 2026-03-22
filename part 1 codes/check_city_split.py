#I first needed to convert the raw dataset into a federated structure, where each city becomes a separate client.
import shutil
from pathlib import Path
#code that verifies the raw files
#organizes data by city
#creates the city-based split
ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data")

CELLMOB = ROOT / "CellMob"
ADDITIONAL = ROOT / "KAUST_additional_modes"

OUT = ROOT / "dataset_by_city"

CITY_DIRS = {
    "kaust": OUT / "kaust",
    "jeddah": OUT / "jeddah",
    "mekkah": OUT / "mekkah",
    "kz": OUT / "kz",
}

for d in CITY_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def copy_csvs(src_folder, dst_folder):
    for file in src_folder.glob("*.csv"):
        
        if file.name.startswith("._"):
            continue
        shutil.copy2(file, dst_folder / file.name)

for folder in CELLMOB.iterdir():

    if not folder.is_dir():
        continue

    name = folder.name.lower()

    if "jeddah" in name:
        copy_csvs(folder, CITY_DIRS["jeddah"])

    elif "mekkah" in name:
        copy_csvs(folder, CITY_DIRS["mekkah"])

    elif "kaust" in name:
        copy_csvs(folder, CITY_DIRS["kaust"])

    elif "_kz" in name:
        copy_csvs(folder, CITY_DIRS["kz"])

for folder in ADDITIONAL.iterdir():

    if folder.is_dir():
        copy_csvs(folder, CITY_DIRS["kaust"])


print("\nDone splitting data by city!")

for city, path in CITY_DIRS.items():
    print(city, "->", len(list(path.glob("*.csv"))), "files")
