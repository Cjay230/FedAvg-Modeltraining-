# PyTorch FedAvg baseline on RAW CSV files (no windowing)
# Input: full data rows, 87 numeric features, 8 classes
# batch_size=512 (batch_size=32 on 6.2M rows = ~193K steps/epoch, impractical on CPU)
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

RAW_DIR     = r"C:\Users\user\Desktop\electives\VIP\kaust data\full data"
SAVE_DIR    = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(SAVE_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'bike':0,'bus':1,'car':2,'jog':3,'motorcycle':4,'scooter':5,'train':6,'walk':7}
CLASS_NAMES = ['bike','bus','car','jog','motorcycle','scooter','train','walk']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 512
LR          = 0.01
MOMENTUM    = 0.9
DROP_COLS   = {'Time','city','mode','source_file'}

torch.manual_seed(42)
np.random.seed(42)

# ── Identify feature columns from all cities ─────────────────────────────────
print("=" * 65)
print("LOADING RAW DATA")
print("=" * 65)

all_feat_cols = set()
for city in CITIES:
    df_sample = pd.read_csv(f'{RAW_DIR}/{city}.csv', nrows=1000, low_memory=False)
    for col in df_sample.columns:
        if col in DROP_COLS or col == 'label':
            continue
        if pd.to_numeric(df_sample[col], errors='coerce').notna().any():
            all_feat_cols.add(col)

feat_cols  = sorted(all_feat_cols)
N_FEATURES = len(feat_cols)
ALL_CLASSES = list(range(N_CLASSES))
print(f"Numeric feature columns: {N_FEATURES}")
print(f"Classes ({N_CLASSES}): {LABEL_MAP}\n")

# ── Load, encode, split ───────────────────────────────────────────────────────
X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}

for city in CITIES:
    print(f"  Loading {city} ...", end=" ", flush=True)
    df = pd.read_csv(f'{RAW_DIR}/{city}.csv', low_memory=False)
    print(f"{len(df):,} rows", end=" | ", flush=True)

    # Coerce all feature columns to numeric, fill NaN -> 0
    for col in feat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    df[feat_cols] = df[feat_cols].fillna(0)

    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].map(LABEL_MAP).values.astype(np.int64)

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)

    X_train_d[city], X_test_d[city] = Xtr, Xte
    y_train_d[city], y_test_d[city] = ytr, yte

    present = sorted(set(y.tolist()))
    print(f"train {len(ytr):,}  test {len(yte):,}  classes={[CLASS_NAMES[c] for c in present]}")

# Global scaler
print("\nFitting StandardScaler ...", end=" ", flush=True)
scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)
X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"done.  Global test set: {len(y_test_global):,} samples\n")

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))

def train_local(global_state, city):
    model = MLP()
    set_state(model, global_state)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
    return model

def fedavg(local_models):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w     = sizes / sizes.sum()
    new_state = {}
    for key in local_models[0].state_dict():
        new_state[key] = sum(
            w[k] * local_models[k].state_dict()[key].float()
            for k in range(len(CITIES))
        )
    return new_state

def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    pc_f1 = f1_score(y_test_global, preds, average=None,
                     labels=ALL_CLASSES, zero_division=0)
    city_m = {}
    for city in CITIES:
        with torch.no_grad():
            yp = model(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        city_m[city] = {
            'acc': float(accuracy_score(y_test_d[city], yp)),
            'f1':  float(f1_score(y_test_d[city], yp, average='macro',
                                  labels=ALL_CLASSES, zero_division=0))
        }
    return g_acc, g_f1, pc_f1, city_m

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FedAvg — PyTorch MLP — RAW data (no windowing)")
print("=" * 65)

global_model = MLP()
global_state = get_state(global_model)
rows = []
t0_exp = time.time()

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    local_models = [train_local(global_state, city) for city in CITIES]
    global_state = fedavg(local_models)
    set_state(global_model, global_state)

    g_acc, g_f1, pc_f1, city_m = evaluate(global_model)
    print(f"  Round {rnd:2d}/10  GlobalAcc {g_acc:.4f}  GlobalMacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'round': rnd, 'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
    for city in CITIES:
        row[f'{city}_acc']      = round(city_m[city]['acc'], 6)
        row[f'{city}_macro_f1'] = round(city_m[city]['f1'],  6)
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, 'fedavg_raw_pytorch.csv')
df.to_csv(out, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
last = df[df['round'] == N_ROUNDS].iloc[0]
total_min = (time.time() - t0_exp) / 60

print(f"\n--- Round 10 Results ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

print(f"\n  Total time: {total_min:.1f} min")
print(f"  Saved -> {out}")

print(f"\n--- Full Comparison ---")
comparison = [
    ("sklearn FedAvg — raw rows (old)",         0.4139, 0.2869),
    ("PyTorch FedAvg — windowed (29 feat, 7cl)",0.7336, 0.4935),
    ("sklearn FedAvg — windowed",               0.8037, 0.6333),
    ("sklearn cwFedAvg+LAWA g=0.2 windowed",    0.8084, 0.6382),
    ("PyTorch cwFedAvg+LAWA g=0.2 windowed",    0.7664, 0.6443),
    ("PyTorch FedAvg — raw (87 feat, 8cl)",     last['global_acc'], last['global_macro_f1']),
]
print(f"  {'Experiment':<46} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*66}")
for name, acc, f1 in comparison:
    print(f"  {name:<46} {acc:>7.4f}  {f1:>8.4f}")
