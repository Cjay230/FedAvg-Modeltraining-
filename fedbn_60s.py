# fedbn_60s.py
# FedBN baseline: federated learning with local batch normalization.
# BN layer parameters are NOT aggregated -- they stay local per client.
# Only non-BN parameters are averaged (FedAvg on shared layers).
# Reference: Li et al., "FedBN: Federated Learning on Non-IID Features
#            via Local Batch Normalization", ICLR 2021.

import os, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

WIN_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(SAVE_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
ALL_CLASSES = list(range(N_CLASSES))
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
print("Loading 60s windowed data...")
city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
N_FEATURES = len(feat_cols)

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}

for city in CITIES:
    df = city_dfs[city].copy()
    df['label_enc'] = df['label'].map(LABEL_MAP).fillna(-1).astype(np.int64)
    df = df[df['label_enc'] >= 0]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_d[city] = Xtr
    X_test_d[city]  = Xte
    y_train_d[city] = ytr.astype(np.int64)
    y_test_d[city]  = yte.astype(np.int64)

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"  Global test: {len(y_test_global)} samples, features: {N_FEATURES}\n")

# ─────────────────────────────────────────────────────────────
# Model with BatchNorm (required for FedBN)
# ─────────────────────────────────────────────────────────────
class MLPwithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc_out(x)

def get_state(m):    return copy.deepcopy(m.state_dict())
def set_state(m, s): m.load_state_dict(copy.deepcopy(s))

def is_bn_key(key):
    return 'bn' in key

# ─────────────────────────────────────────────────────────────
# Aggregation: skip BN layers (keep them local)
# ─────────────────────────────────────────────────────────────
def aggregate_fedbn(local_states, local_bn_states, sizes):
    """Average non-BN parameters by sample count; restore per-client BN."""
    total = sum(sizes)
    w     = [s / total for s in sizes]

    global_state = {}
    ref_keys = local_states[0].keys()
    for key in ref_keys:
        if is_bn_key(key):
            # BN keys are not aggregated — will be restored per client
            global_state[key] = local_states[0][key]
        else:
            global_state[key] = sum(w[k] * local_states[k][key].float()
                                    for k in range(len(local_states)))
    return global_state

# ─────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────
def evaluate(global_state, client_bn_states):
    """Evaluate by averaging city predictions (each city uses its own BN)."""
    all_preds = []
    all_labels = []
    for k, city in enumerate(CITIES):
        state = copy.deepcopy(global_state)
        for key in client_bn_states[k]:
            state[key] = client_bn_states[k][key]
        m = MLPwithBN()
        set_state(m, state)
        m.eval()
        with torch.no_grad():
            X = torch.tensor(X_test_d[city])
            if X.shape[0] == 1:
                m.train()  # BN needs >1 sample; fallback
            preds = m(X).argmax(dim=1).numpy()
        all_preds.append(preds)
        all_labels.append(y_test_d[city])
    preds_cat  = np.concatenate(all_preds)
    labels_cat = np.concatenate(all_labels)
    g_acc = float(accuracy_score(labels_cat, preds_cat))
    g_f1  = float(f1_score(labels_cat, preds_cat, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    city_f1 = {}
    for k, city in enumerate(CITIES):
        city_f1[city] = float(f1_score(all_labels[k], all_preds[k],
                                        average='macro', labels=ALL_CLASSES,
                                        zero_division=0))
    return g_acc, g_f1, city_f1

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_local(global_state, client_bn_state, city):
    """Standard local training; BN parameters restored from client state."""
    state = copy.deepcopy(global_state)
    for key in client_bn_state:
        state[key] = client_bn_state[key]

    model = MLPwithBN()
    set_state(model, state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader    = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)

    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    full_state = get_state(model)
    bn_state   = {k: v for k, v in full_state.items() if is_bn_key(k)}
    return full_state, bn_state

# ─────────────────────────────────────────────────────────────
# FL loop
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("FedBN Baseline -- 60s windows, 10 rounds")
print("BN layers kept local; only shared layers aggregated by FedAvg")
print("=" * 65)

global_model     = MLPwithBN()
global_state     = get_state(global_model)
client_bn_states = [{k: v for k, v in global_state.items() if is_bn_key(k)}
                    for _ in CITIES]

rows   = []
t0_exp = time.time()

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    local_full_states = []
    new_bn_states     = []
    sizes             = []

    for k, city in enumerate(CITIES):
        full_s, bn_s = train_local(global_state, client_bn_states[k], city)
        local_full_states.append(full_s)
        new_bn_states.append(bn_s)
        sizes.append(len(y_train_d[city]))

    global_state     = aggregate_fedbn(local_full_states, new_bn_states,
                                        sizes)
    client_bn_states = new_bn_states

    g_acc, g_f1, city_f1 = evaluate(global_state, client_bn_states)
    print(f"  Round {rnd:2d}/{N_ROUNDS}  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'round': rnd, 'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
    for city in CITIES:
        row[f'{city}_macro_f1'] = round(city_f1[city], 6)
    rows.append(row)

df  = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, 'fedbn_60s.csv')
df.to_csv(out, index=False)

last      = df[df['round'] == N_ROUNDS].iloc[0]
total_min = (time.time() - t0_exp) / 60

print(f"\n  Done in {total_min:.1f} min  ->  {out}")
print(f"\n--- Final Results (Round {N_ROUNDS}) ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")

print(f"\n--- Comparison ---")
comp = [
    ("SCAFFOLD (vanilla SGD)",          0.7163, 0.3663),
    ("FedProx (mu=0.01)",               0.7827, 0.5220),
    ("FedAvg baseline (E1)",            0.7837, 0.5243),
    ("FedBN (this run)",                last['global_acc'], last['global_macro_f1']),
    ("FedPer",                          0.8421, 0.5845),
    ("Centralized plain CE",            0.8421, 0.5908),
    ("E5: full proposed (federated)",   0.8048, 0.6313),
]
print(f"  {'Method':<40} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*60}")
for name, acc, f1 in comp:
    print(f"  {name:<40} {float(acc):>7.4f}  {float(f1):>8.4f}")
print("Done.")
