import os
import pandas as pd
import numpy as np

# =============================
# PATHS
# =============================

dataset_path = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\kaust_multiclass_dataset_clean.csv"

clients_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\clients"

os.makedirs(clients_dir, exist_ok=True)

# number of simulated clients
N_CLIENTS = 5

# =============================
# LOAD DATASET
# =============================

print("Loading KAUST dataset...")
df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)

# =============================
# SHUFFLE DATASET
# =============================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =============================
# SPLIT INTO CLIENTS
# =============================

client_splits = np.array_split(df, N_CLIENTS)

print("\nCreating client datasets...\n")

for i, split in enumerate(client_splits):

    client_id = i + 1

    # convert split back to dataframe
    client_df = pd.DataFrame(split, columns=df.columns)

    save_path = os.path.join(
        clients_dir,
        f"client_{client_id}.csv"
    )

    client_df.to_csv(save_path, index=False)

    print(
        f"Client {client_id}: {client_df.shape[0]} samples saved"
    )

print("\nAll client datasets created successfully.")