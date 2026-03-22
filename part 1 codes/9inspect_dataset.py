import pandas as pd
from pathlib import Path
#I used this to inspect the final training files and confirm that the prepared federated datasets had the right columns and structure before training
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