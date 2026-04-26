# FedPer: Federated Personalization on 60s windowed data
# Shared base  = hidden layers (net: input->256->256) federated via FedAvg
# Personal head = output layer (fc_out: 256->N_CLASSES) kept local, never shared
#
# Each round:
#   1. Broadcast global base weights to all clients
#   2. Each client loads global base + its own personal head, trains locally
#   3. Server aggregates ONLY the base (hidden layers) via FedAvg
#   4. Each client's personal head is updated locally and retained
#
# Global evaluation: concatenate per-client predictions (each uses its own head)
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
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
ALL_CLASSES = list(range(N_CLASSES))

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# LOAD 60s WINDOWED DATA
# -----------------------------------------------------------------------------
print("=" * 65)
print("FedPer -- 60s windowed data")
print("Shared base (hidden layers) + personalized head (fc_out) per city")
print("=" * 65)

city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))
    print(f"  {city:8s}: {len(city_dfs[city])} windows")

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
N_FEATURES = len(feat_cols)
print(f"  Features: {N_FEATURES}\n")

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
city_classes = {}

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
    X_train_d[city] = Xtr
    X_test_d[city]  = Xte
    y_train_d[city] = ytr.astype(np.int64)
    y_test_d[city]  = yte.astype(np.int64)
    city_classes[city] = set(int(c) for c in np.unique(ytr))
    present = [CLASS_NAMES[c] for c in sorted(city_classes[city])]
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  classes={present}")

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test: {len(y_test_global)} samples")

# -----------------------------------------------------------------------------
# MODEL  (base + head clearly separated)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(            # shared base
            nn.Linear(N_FEATURES, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, N_CLASSES)   # personalized head

    def forward(self, x):
        return self.fc_out(self.net(x))

def base_keys(state_dict):
    return [k for k in state_dict if k.startswith('net.')]

def head_keys(state_dict):
    return [k for k in state_dict if k.startswith('fc_out.')]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def get_state(m):    return copy.deepcopy(m.state_dict())
def set_state(m, s): m.load_state_dict(copy.deepcopy(s))

def apply_base(model, base_state):
    """Load only the base (hidden) layers from base_state into model."""
    sd = model.state_dict()
    for k in base_keys(base_state):
        sd[k] = base_state[k].clone()
    model.load_state_dict(sd)

def extract_base(state_dict):
    return {k: v.clone() for k, v in state_dict.items() if k.startswith('net.')}

def fedavg_base(local_states):
    """Aggregate only the base layers, weighted by training sample count."""
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_base = {}
    keys = base_keys(local_states[0])
    for key in keys:
        new_base[key] = sum(w[k] * local_states[k][key].float()
                            for k in range(len(CITIES)))
    return new_base

# -----------------------------------------------------------------------------
# LOCAL TRAINING
# -----------------------------------------------------------------------------
def train_local(global_base, personal_head_state, city):
    """
    Load global base + city's personal head, train for one epoch,
    return updated full state (base updated by SGD, head updated by SGD).
    """
    model = MLP()
    # Start from global base
    apply_base(model, global_base)
    # Load city's own personal head
    sd = model.state_dict()
    for k in head_keys(personal_head_state):
        sd[k] = personal_head_state[k].clone()
    model.load_state_dict(sd)

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

    return get_state(model)

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(global_base, personal_heads):
    """
    Global metrics: concatenate per-client predictions using each city's
    personal head applied on top of the global base.
    Per-city metrics: each city evaluates on its own test set with its own head.
    """
    all_preds, all_true = [], []
    city_m = {}

    for city in CITIES:
        model = MLP()
        apply_base(model, global_base)
        sd = model.state_dict()
        for k in head_keys(personal_heads[city]):
            sd[k] = personal_heads[city][k].clone()
        model.load_state_dict(sd)
        model.eval()

        with torch.no_grad():
            yp = model(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        all_preds.append(yp)
        all_true.append(y_test_d[city])

        city_m[city] = float(f1_score(y_test_d[city], yp, average='macro',
                                      labels=ALL_CLASSES, zero_division=0))

    preds_cat = np.concatenate(all_preds)
    true_cat  = np.concatenate(all_true)
    g_acc = float(accuracy_score(true_cat, preds_cat))
    g_f1  = float(f1_score(true_cat, preds_cat, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    pc_f1 = f1_score(true_cat, preds_cat, average=None,
                     labels=ALL_CLASSES, zero_division=0)
    return g_acc, g_f1, pc_f1, city_m

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
print(f"\n{'='*65}")
print("  Training FedPer (10 rounds)")
print(f"{'='*65}")

# Initialise global base and per-city personal heads
init_model   = MLP()
global_base  = extract_base(get_state(init_model))

# Each city starts with a fresh random head
personal_heads = {city: {k: v.clone()
                          for k, v in get_state(MLP()).items()
                          if k.startswith('fc_out.')}
                  for city in CITIES}

rows   = []
t0_exp = time.time()

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()

    # Local training — each city trains base + its own head
    local_states = []
    for city in CITIES:
        state = train_local(global_base, personal_heads[city], city)
        local_states.append(state)
        # Update city's personal head immediately
        personal_heads[city] = {k: state[k].clone()
                                 for k in head_keys(state)}

    # Aggregate only the base layers
    global_base = fedavg_base(local_states)

    g_acc, g_f1, pc_f1, city_m = evaluate(global_base, personal_heads)
    print(f"  Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'round': rnd,
           'global_acc':      round(g_acc, 6),
           'global_macro_f1': round(g_f1,  6)}
    for city in CITIES:
        row[f'{city}_macro_f1'] = round(city_m[city], 6)
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    rows.append(row)

df  = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, 'fedper_60s.csv')
df.to_csv(out, index=False)

last      = df[df['round'] == N_ROUNDS].iloc[0]
total_min = (time.time() - t0_exp) / 60

print(f"\n  Done in {total_min:.1f} min  ->  {out}")
print(f"\n--- Round 10 Results ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

# -----------------------------------------------------------------------------
# COMPARISON TABLE
# -----------------------------------------------------------------------------
print(f"\n--- Comparison (Round 10, 60s windows) ---")
comp = [
    ("FedAvg baseline (E1)",          0.7837, 0.5243, 0.2989, 0.5412, 0.1554, 0.2324),
    ("E5: full proposed method",      0.8048, 0.6313, 0.3013, 0.6417, 0.1957, 0.2387),
    ("Centralized (no FL)",           0.8421, 0.5908, 0.3278, 0.5855, 0.2305, 0.2538),
    ("FedProx (mu=0.01)",             0.7827, 0.5220, 0.2989, 0.5384, 0.1554, 0.2351),
    ("SCAFFOLD (momentum=0)",         0.7163, 0.3663, 0.2748, 0.3459, 0.1203, 0.2296),
    ("FedPer (this run)",
     last['global_acc'], last['global_macro_f1'],
     last['jeddah_macro_f1'], last['kaust_macro_f1'],
     last['kz_macro_f1'],    last['mekkah_macro_f1']),
]
print(f"\n  {'Experiment':<32} {'Acc':>7}  {'MacroF1':>8}  {'jeddah':>8}  {'kaust':>7}  {'kz':>6}  {'mekkah':>8}")
print(f"  {'-'*82}")
for row in comp:
    name = row[0]
    vals = [float(v) for v in row[1:]]
    print(f"  {name:<32} {vals[0]:>7.4f}  {vals[1]:>8.4f}  {vals[2]:>8.4f}  {vals[3]:>7.4f}  {vals[4]:>6.4f}  {vals[5]:>8.4f}")
print("Done.")
