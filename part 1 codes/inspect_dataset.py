import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city\clients")

for f in ROOT.glob("*.csv"):
    print("\n==============================")
    print("FILE:", f.name)

    df = pd.read_csv(f, nrows=5, low_memory=False)

    print("\nColumns:")
    for c in df.columns:
        print("-", c)

    print("\nShape preview:", df.shape)
    print("\nSample rows:")
    print(df.head())