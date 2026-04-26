# gamma_sweep_clean.py
# Controlled gamma sweep: seed=42 reset before EVERY gamma run.
# All 6 gammas start from identical conditions (same data split, same model init).
# gamma=0.0 => pure cwFedAvg (LAWA weight=0), gamma=0.5 => heavy LAWA.
# Everything else is the full E5 setup: CW loss + KI + cwFedAvg + LAWA.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIN_DIR   = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
PLOTS_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\plots"
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
ALL_CLASSES = list(range(N_CLASSES))

# ─────────────────────────────────────────────────────────────
# Data loaded ONCE with fixed random_state=42 (deterministic split).
# All gamma runs share the same train/test split.
# ─────────────────────────────────────────────────────────────
print("Loading 60s windowed data...")
city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
N_FEATURES = len(feat_cols)

X_train_d, X_test_d, y_train_d, y_test_d, city_classes = {}, {}, {}, {}, {}
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
    city_classes[city] = set(int(c) for c in np.unique(ytr))

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"  Global test: {len(y_test_global)} samples, features: {N_FEATURES}\n")

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
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

def get_state(m):    return copy.deepcopy(m.state_dict())
def set_state(m, s): m.load_state_dict(copy.deepcopy(s))

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def class_count_vec(y):
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        v[c] = (y == c).sum()
    return v

def evaluate(global_state):
    m = MLP()
    set_state(m, global_state)
    m.eval()
    with torch.no_grad():
        preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    city_m = {}
    for city in CITIES:
        with torch.no_grad():
            yp = m(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        city_m[city] = float(f1_score(y_test_d[city], yp, average='macro',
                                       labels=ALL_CLASSES, zero_division=0))
    return g_acc, g_f1, city_m

def train_local(global_state, city):
    """Full E5 local training: CW loss + knowledge inheritance."""
    model = MLP()
    set_state(model, global_state)
    model.train()

    y_arr   = y_train_d[city]
    n_tot   = len(y_arr)
    weights = torch.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        n_c = (y_arr == c).sum()
        if n_c > 0:
            weights[c] = n_tot / (N_CLASSES * n_c)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader    = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)

    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    # Knowledge inheritance: freeze absent-class output nodes
    missing = [c for c in range(N_CLASSES) if c not in city_classes[city]]
    if missing:
        with torch.no_grad():
            for c in missing:
                model.fc_out.weight[c].copy_(global_state['fc_out.weight'][c])
                model.fc_out.bias[c].copy_(global_state['fc_out.bias'][c])
    return model

def compute_lawa_weights(local_models):
    n     = len(local_models)
    raw_w = np.zeros(n)
    for k, model in enumerate(local_models):
        city = CITIES[k]
        model.eval()
        with torch.no_grad():
            log_prob = torch.log_softmax(
                model(torch.tensor(X_train_d[city])), dim=1).numpy()
        y      = y_train_d[city]
        losses = [-log_prob[(y == c), c].mean()
                  for c in range(N_CLASSES) if (y == c).sum() > 0]
        if len(losses) < 2:
            raw_w[k] = 1.0
        else:
            raw_w[k] = min(losses) / max(losses) if max(losses) > 0 else 1.0
    s = raw_w.sum()
    return raw_w / s if s > 0 else np.ones(n) / n

def aggregate(local_models, gamma):
    """cwFedAvg + LAWA blend at the given gamma.
       gamma=0.0 => pure cwFedAvg (LAWA contributes nothing).
       gamma=1.0 => pure LAWA.
    """
    n    = len(local_models)
    cc   = np.array([class_count_vec(y_train_d[c]) for c in CITIES])
    sizes = cc.sum(axis=1)
    w_size = sizes / sizes.sum()               # sample-proportional (hidden layers)
    ct     = cc.sum(axis=0)
    w_cw   = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)  # per-class

    lawa   = compute_lawa_weights(local_models)

    # Blended hidden-layer weights
    w_h = gamma * lawa + (1 - gamma) * w_size
    w_h /= w_h.sum()

    # Blended per-class output weights
    w_o = gamma * lawa[:, None] + (1 - gamma) * w_cw   # (n, C)
    for c in range(N_CLASSES):
        s = w_o[:, c].sum()
        if s > 0:
            w_o[:, c] /= s

    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(w_h[k] * local_models[k].state_dict()[key].float()
                             for k in range(n))

    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += w_o[k,c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += w_o[k,c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state

# ─────────────────────────────────────────────────────────────
# Gamma sweep — seed=42 reset before EVERY run
# ─────────────────────────────────────────────────────────────
gammas  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
f1_vals = []
acc_vals = []

print("=" * 65)
print("GAMMA SWEEP (seed=42 reset per run, identical init for all gammas)")
print("Full E5: CW loss + Knowledge Inheritance + cwFedAvg + LAWA")
print("=" * 65)

for gamma in gammas:
    torch.manual_seed(42)
    np.random.seed(42)

    global_state = get_state(MLP())

    for rnd in range(1, N_ROUNDS + 1):
        local_models = [train_local(global_state, city) for city in CITIES]
        global_state = aggregate(local_models, gamma)

    g_acc, g_f1, city_m = evaluate(global_state)
    f1_vals.append(g_f1)
    acc_vals.append(g_acc)
    print(f"  gamma={gamma:.1f}  Acc={g_acc:.4f}  Macro F1={g_f1:.4f}")

# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────
lines = [
    "Gamma Sweep — Full E5 (CW loss + KI + cwFedAvg + LAWA)\n",
    "seed=42 reset before each run — all gammas use identical init\n",
    f"\n{'gamma':>8}  {'Accuracy':>10}  {'Macro F1':>10}\n",
    "-"*34 + "\n",
]
for g, a, f in zip(gammas, acc_vals, f1_vals):
    lines.append(f"{g:>8.1f}  {a:>10.4f}  {f:>10.4f}\n")
with open(os.path.join(SAVE_DIR, "gamma_sweep.txt"), "w") as fh:
    fh.writelines(lines)

# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(gammas, f1_vals, marker='o', linewidth=2, markersize=8,
        color='steelblue', label='E5 method')
ax.set_xlabel('Gamma (γ)', fontsize=12)
ax.set_ylabel('Global Macro F1 (Round 10)', fontsize=12)
ax.set_title('Macro F1 vs. Gamma — E5 Method, 60s Windows', fontsize=13)
ax.set_xticks(gammas)
ax.set_ylim(min(f1_vals) - 0.02, max(f1_vals) + 0.03)
ax.grid(True, alpha=0.4)
ax.legend()
for g, f in zip(gammas, f1_vals):
    ax.annotate(f'{f:.4f}', (g, f), textcoords='offset points',
                xytext=(0, 9), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_gamma_sweep.png"), dpi=150)
plt.close()

print(f"\nSaved: results/gamma_sweep.txt, plots/11_gamma_sweep.png")
