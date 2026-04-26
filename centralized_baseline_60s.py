# Centralized baseline: pool all 4 cities, train single MLP, no federated learning
# 60s windowed data, same architecture (input->256->256->N_CLASSES) and split logic
import os, time
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
N_EPOCHS    = 10       # same number of passes as federated rounds
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# LOAD & POOL 60s WINDOWED DATA
# -----------------------------------------------------------------------------
print("=" * 65)
print("Centralized Baseline -- 60s windowed data, pooled")
print("=" * 65)

city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))
    print(f"  {city:8s}: {len(city_dfs[city])} windows")

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
N_FEATURES = len(feat_cols)
print(f"  Features: {N_FEATURES}\n")

# Per-city stratified 80/20 split (same logic as federated experiments),
# then pool train sets and pool test sets separately
X_train_parts, X_test_parts = [], []
y_train_parts, y_test_parts = [], []
X_test_d, y_test_d = {}, {}

for city in CITIES:
    df = city_dfs[city].copy()
    df['label_enc'] = df['label'].map(LABEL_MAP).fillna(-1).astype(np.int64)
    df = df[df['label_enc'] >= 0]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)
    X_train_parts.append(Xtr)
    X_test_parts.append(Xte)
    y_train_parts.append(ytr)
    y_test_parts.append(yte)
    X_test_d[city] = Xte
    y_test_d[city] = yte.astype(np.int64)
    present = sorted(set(y.tolist()))
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  "
          f"classes={[CLASS_NAMES[c] for c in present]}")

X_train = np.vstack(X_train_parts).astype(np.float32)
X_test  = np.vstack(X_test_parts).astype(np.float32)
y_train = np.concatenate(y_train_parts).astype(np.int64)
y_test  = np.concatenate(y_test_parts).astype(np.int64)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
for city in CITIES:
    X_test_d[city] = scaler.transform(X_test_d[city]).astype(np.float32)

print(f"\n  Pooled train: {len(y_train):,}  |  Global test: {len(y_test):,} samples")

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, N_CLASSES)
    def forward(self, x):
        return self.fc_out(self.net(x))

model     = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
loader    = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Training ({N_EPOCHS} epochs)")
print(f"{'='*65}")

rows  = []
t0_exp = time.time()

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    model.train()
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test, preds))
    g_f1  = float(f1_score(y_test, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    pc_f1 = f1_score(y_test, preds, average=None,
                     labels=ALL_CLASSES, zero_division=0)
    print(f"  Epoch {epoch:2d}/{N_EPOCHS}  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'epoch': epoch, 'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    for city in CITIES:
        with torch.no_grad():
            yp = model(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        row[f'{city}_macro_f1'] = round(float(f1_score(
            y_test_d[city], yp, average='macro',
            labels=ALL_CLASSES, zero_division=0)), 6)
    rows.append(row)

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
df  = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, 'centralized_baseline_60s.csv')
df.to_csv(out, index=False)

last      = df[df['epoch'] == N_EPOCHS].iloc[0]
total_min = (time.time() - t0_exp) / 60

print(f"\n  Done in {total_min:.1f} min  ->  {out}")
print(f"\n--- Final Results (Epoch {N_EPOCHS}) ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

print(f"\n--- Comparison vs Federated ---")
comp = [
    ("FedAvg baseline (E1, federated)",          0.7837, 0.5243),
    ("E5: full proposed (federated)",            0.8048, 0.6313),
    ("Centralized (this run)",                   last['global_acc'], last['global_macro_f1']),
]
print(f"  {'Experiment':<42} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*62}")
for name, acc, f1 in comp:
    print(f"  {name:<42} {float(acc):>7.4f}  {float(f1):>8.4f}")
print("Done.")
