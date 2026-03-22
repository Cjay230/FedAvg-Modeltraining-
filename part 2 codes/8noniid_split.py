import os
import pandas as pd
import numpy as np
#Each non-IID client trains its own local model, and then their predictions are aggregated using majority voting
# =============================
# PATHS
# =============================

dataset_path = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\kaust_multiclass_dataset_clean.csv"

save_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\noniid_clients"

os.makedirs(save_dir, exist_ok=True)

# =============================
# LOAD DATA
# =============================

print("Loading KAUST dataset...")
df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =============================
# DEFINE CLIENT LABEL SKEW
# =============================

client_modes = {
    "client_1": ["walk", "jog"],
    "client_2": ["bike", "scooter"],
    "client_3": ["car", "motorcycle"],
    "client_4": ["bus_colored", "bus_ondemand"],
    "client_5": ["walk", "car", "bike"]
}

clients = {k: [] for k in client_modes.keys()}

# =============================
# DISTRIBUTE DATA
# =============================

for label, group in df.groupby("label"):

    group = group.sample(frac=1, random_state=42)

    n = len(group)

    for client, modes in client_modes.items():

        if label in modes:
            portion = 0.6
        else:
            portion = 0.1

        size = int(n * portion / len(client_modes))

        clients[client].append(group.iloc[:size])

# =============================
# SAVE CLIENT DATASETS
# =============================

for client, parts in clients.items():

    client_df = pd.concat(parts).sample(frac=1).reset_index(drop=True)

    path = os.path.join(save_dir, f"{client}.csv")

    client_df.to_csv(path, index=False)

    print(f"{client} -> {client_df.shape}")

print("\nNon-IID clients created successfully.")