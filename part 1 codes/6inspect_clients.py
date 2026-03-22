import pandas as pd
from pathlib import Path
#Before training, I inspected the resulting client datasets to confirm that each city file had the expected structure and labels.
ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city\clients")

for f in ROOT.glob("*.csv"):
    print("\n" + "="*60)
    print("FILE:", f.name)
    df = pd.read_csv(f, nrows=5, low_memory=False)
    print("Shape preview:", df.shape)
    print("Columns:")
    for c in df.columns:
        print("-", c)
    print("\nFirst 5 rows:")
    print(df.head())