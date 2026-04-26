# fedbn_perclass_check.py
# Reruns FedBN and prints per-class F1 at round 10, including jog.
# Run this once to get the number needed for the paper argument.

import os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

WIN_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
ALL_CLASSES = list(range(N_CLASSES))
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR, MOMENTUM = 0.01, 0.9

torch.manual_seed(42)
np.random.seed(42)

print("Loading data...")
city_dfs = {c: pd.read_csv(os.path.join(WIN_DIR, f'{c}_60s.csv')) for c in CITIES}
feat_cols = [col for col in city_dfs['jeddah'].columns if col != 'label']
N_FEATURES = len(feat_cols)

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
for city in CITIES:
    df = city_dfs[city].copy()
    df['label_enc'] = df['label'].map(LABEL_MAP).fillna(-1).astype(np.int64)
    df = df[df['label_enc'] >= 0]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_d[city], X_test_d[city] = Xtr, Xte
    y_train_d[city], y_test_d[city] = ytr.astype(np.int64), yte.astype(np.int64)

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

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
        return self.fc_out(self.relu(self.bn2(self.fc2(self.relu(self.bn1(self.fc1(x)))))))

def is_bn_key(key): return 'bn' in key
def get_state(m): return copy.deepcopy(m.state_dict())
def set_state(m, s): m.load_state_dict(copy.deepcopy(s))

def aggregate_fedbn(local_states, sizes):
    total = sum(sizes)
    w = [s / total for s in sizes]
    global_state = {}
    for key in local_states[0].keys():
        if is_bn_key(key):
            global_state[key] = local_states[0][key]
        else:
            global_state[key] = sum(w[k] * local_states[k][key].float() for k in range(len(local_states)))
    return global_state

def train_local(global_state, client_bn_state, city):
    state = copy.deepcopy(global_state)
    for key in client_bn_state:
        state[key] = client_bn_state[key]
    model = MLPwithBN()
    set_state(model, state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]), torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
    full_state = get_state(model)
    bn_state = {k: v for k, v in full_state.items() if is_bn_key(k)}
    return full_state, bn_state

def get_all_preds(global_state, client_bn_states):
    all_preds, all_labels = [], []
    for k, city in enumerate(CITIES):
        state = copy.deepcopy(global_state)
        for key in client_bn_states[k]:
            state[key] = client_bn_states[k][key]
        m = MLPwithBN()
        set_state(m, state)
        m.eval()
        with torch.no_grad():
            X = torch.tensor(X_test_d[city])
            preds = m(X).argmax(dim=1).numpy()
        all_preds.append(preds)
        all_labels.append(y_test_d[city])
    return np.concatenate(all_preds), np.concatenate(all_labels)

# Run FL
global_model = MLPwithBN()
global_state = get_state(global_model)
client_bn_states = [{k: v for k, v in global_state.items() if is_bn_key(k)} for _ in CITIES]

print("Running FedBN (10 rounds)...")
for rnd in range(1, N_ROUNDS + 1):
    local_full_states, new_bn_states, sizes = [], [], []
    for k, city in enumerate(CITIES):
        full_s, bn_s = train_local(global_state, client_bn_states[k], city)
        local_full_states.append(full_s)
        new_bn_states.append(bn_s)
        sizes.append(len(y_train_d[city]))
    global_state = aggregate_fedbn(local_full_states, sizes)
    client_bn_states = new_bn_states
    preds, labels = get_all_preds(global_state, client_bn_states)
    macro_f1 = f1_score(labels, preds, average='macro', labels=ALL_CLASSES, zero_division=0)
    print(f"  Round {rnd:2d}  MacroF1={macro_f1:.4f}")

# Final per-class breakdown
preds, labels = get_all_preds(global_state, client_bn_states)
macro_f1 = f1_score(labels, preds, average='macro', labels=ALL_CLASSES, zero_division=0)

print("\n" + "="*60)
print(f"FedBN Final Results — Round {N_ROUNDS}")
print(f"Global Macro F1: {macro_f1:.4f}")
print("="*60)
print("\nPer-class F1 (all classes):")
report = classification_report(labels, preds, labels=ALL_CLASSES,
                                target_names=CLASS_NAMES, zero_division=0)
print(report)

# Pull out jog specifically
jog_idx = LABEL_MAP['jog']
jog_f1 = f1_score(labels, preds, labels=[jog_idx], average='macro', zero_division=0)
print(f"\n>>> JOG F1 (FedBN): {jog_f1:.4f}")
print(f">>> JOG F1 (E5):    0.5670")
print(f">>> JOG F1 (FedAvg): 0.0000")
print()

# Also check motorcycle
moto_idx = LABEL_MAP['motorcycle']
moto_f1 = f1_score(labels, preds, labels=[moto_idx], average='macro', zero_division=0)
print(f">>> MOTORCYCLE F1 (FedBN): {moto_f1:.4f}")
print(f">>> MOTORCYCLE F1 (E5):    0.5080")
print(f">>> MOTORCYCLE F1 (FedAvg): 0.4000")
print("\nDone. Use jog F1 number in the paper argument.")
