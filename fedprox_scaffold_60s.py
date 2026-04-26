# FedProx (mu=0.01) and SCAFFOLD on 60s windowed data
# Same MLP (256-256), SGD lr=0.01 momentum=0.9, batch_size=32, 10 rounds
# as the ablation and window comparison experiments.
#
# FedProx:  local loss = CE + (mu/2) * ||w_local - w_global||^2
# SCAFFOLD: gradient corrected with control variates to reduce client drift
#           c_i_new = c_i - c + (w_global - w_local) / (K * lr)
#           c_new   = c + (1/n) * sum(delta_c_i)
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
MU          = 0.01   # FedProx proximal coefficient
ALL_CLASSES = list(range(N_CLASSES))

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# LOAD 60s WINDOWED CACHE
# -----------------------------------------------------------------------------
print("=" * 65)
print("LOADING 60s WINDOWED DATA")
print("=" * 65)

city_dfs = {}
for city in CITIES:
    path = os.path.join(WIN_DIR, f'{city}_60s.csv')
    city_dfs[city] = pd.read_csv(path)
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

def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))

def zero_state(model):
    """Return a state dict of zeros matching the model's parameter shapes."""
    return {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(global_state):
    m = MLP()
    set_state(m, global_state)
    m.eval()
    with torch.no_grad():
        preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    pc_f1 = f1_score(y_test_global, preds, average=None,
                     labels=ALL_CLASSES, zero_division=0)
    city_m = {}
    for city in CITIES:
        with torch.no_grad():
            yp = m(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        city_m[city] = float(f1_score(y_test_d[city], yp, average='macro',
                                      labels=ALL_CLASSES, zero_division=0))
    return g_acc, g_f1, pc_f1, city_m

# -----------------------------------------------------------------------------
# AGGREGATION  (standard FedAvg weighted by sample count)
# -----------------------------------------------------------------------------
def fedavg_agg(local_states):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_state = {}
    for key in local_states[0]:
        new_state[key] = sum(w[k] * local_states[k][key].float()
                             for k in range(len(CITIES)))
    return new_state

# -----------------------------------------------------------------------------
# FEDPROX LOCAL TRAINING
# -----------------------------------------------------------------------------
def train_fedprox(global_state, city, mu=MU):
    model = MLP()
    set_state(model, global_state)

    # Freeze a copy of the global weights for the proximal term
    global_params = {k: v.clone().detach() for k, v in model.named_parameters()}

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)

    for Xb, yb in loader:
        optimizer.zero_grad()
        ce_loss = criterion(model(Xb), yb)

        # Proximal term: (mu/2) * ||w - w_global||^2
        prox = torch.tensor(0.0)
        for name, param in model.named_parameters():
            prox = prox + ((param - global_params[name]) ** 2).sum()
        loss = ce_loss + (mu / 2.0) * prox

        loss.backward()
        optimizer.step()

    return get_state(model)

# -----------------------------------------------------------------------------
# SCAFFOLD LOCAL TRAINING
# -----------------------------------------------------------------------------
def train_scaffold(global_state, city, c_global, c_local):
    """
    c_global : dict {key -> tensor}  server control variate
    c_local  : dict {key -> tensor}  this client's control variate
    Returns (new_local_state, delta_c_local, K)
    """
    model = MLP()
    set_state(model, global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)

    K = 0  # count gradient steps
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()

        # Apply control-variate correction: grad += c_global - c_local
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.add_(c_global[name] - c_local[name])

        optimizer.step()
        K += 1

    new_state = get_state(model)

    # Update client control variate (Option II):
    # c_i_new = c_i - c + (w_global - w_local) / (K * lr)
    new_c_local = {}
    delta_c     = {}
    for key in global_state:
        correction = (global_state[key].float() - new_state[key].float()) / (K * LR)
        new_c_local[key] = c_local[key] + correction - c_global[key]
        delta_c[key]     = new_c_local[key] - c_local[key]

    return new_state, new_c_local, delta_c

# -----------------------------------------------------------------------------
# GENERIC TRAINING LOOP
# -----------------------------------------------------------------------------
def run_experiment(name, save_name, round_fn):
    """
    round_fn(rnd) -> (global_state_new, global_state_new)
    Caller is responsible for maintaining experiment state via closure.
    """
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

    rows = []
    t0_exp = time.time()

    global_state = get_state(MLP())

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        global_state = round_fn(global_state, rnd)
        g_acc, g_f1, pc_f1, city_m = evaluate(global_state)
        print(f"  Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

        row = {'experiment': name, 'round': rnd,
               'global_acc': round(g_acc, 6), 'global_macro_f1': round(g_f1, 6)}
        for city in CITIES:
            row[f'{city}_macro_f1'] = round(city_m[city], 6)
        for i, cls in enumerate(CLASS_NAMES):
            row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, save_name)
    df.to_csv(out, index=False)

    last = df[df['round'] == N_ROUNDS].iloc[0]
    total_min = (time.time() - t0_exp) / 60
    print(f"  Done in {total_min:.1f} min  ->  {out}")
    print(f"\n  --- Round 10 Results ---")
    print(f"  Global Accuracy : {last['global_acc']:.4f}")
    print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
    print(f"  Per-city Macro F1:")
    for city in CITIES:
        print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")
    return df

# =============================================================================
# EXPERIMENT 1: FedProx (mu=0.01)
# =============================================================================
def make_fedprox_round():
    def round_fn(global_state, rnd):
        local_states = [train_fedprox(global_state, city) for city in CITIES]
        return fedavg_agg(local_states)
    return round_fn

df_fedprox = run_experiment(
    name      = f"FedProx (mu={MU})",
    save_name = "fedprox_60s.csv",
    round_fn  = make_fedprox_round(),
)

# =============================================================================
# EXPERIMENT 2: SCAFFOLD
# =============================================================================
def make_scaffold_round():
    ref = MLP()
    # Server control variate (initialised to zeros)
    c_global = zero_state(ref)
    # Per-client control variates (initialised to zeros)
    c_locals  = {city: zero_state(ref) for city in CITIES}

    def round_fn(global_state, rnd):
        local_states = []
        delta_cs     = []

        for city in CITIES:
            new_state, new_c_local, delta_c = train_scaffold(
                global_state, city, c_global, c_locals[city])
            local_states.append(new_state)
            delta_cs.append(delta_c)
            c_locals[city] = new_c_local   # update in-place for next round

        # Aggregate model
        new_global = fedavg_agg(local_states)

        # Update server control variate: c += (1/n) * sum(delta_c_i)
        n = len(CITIES)
        for key in c_global:
            c_global[key] = c_global[key] + sum(dc[key] for dc in delta_cs) / n

        return new_global

    return round_fn

df_scaffold = run_experiment(
    name      = "SCAFFOLD",
    save_name = "scaffold_60s.csv",
    round_fn  = make_scaffold_round(),
)

# =============================================================================
# COMBINED SAVE + COMPARISON TABLE
# =============================================================================
combined = pd.concat([df_fedprox, df_scaffold], ignore_index=True)
combined.to_csv(os.path.join(SAVE_DIR, "fedprox_scaffold_60s_all.csv"), index=False)

last_rows = combined[combined['round'] == N_ROUNDS].copy()

# Add FedAvg baseline from ablation for reference
fedavg_ref = {
    'experiment': 'FedAvg baseline (ref)',
    'global_acc': 0.7837, 'global_macro_f1': 0.5243,
    'jeddah_macro_f1': 0.2989, 'kaust_macro_f1': 0.5412,
    'kz_macro_f1': 0.1554, 'mekkah_macro_f1': 0.2324,
    'f1_car': 0.6433, 'f1_walk': 0.9425, 'f1_bus': 0.7159,
    'f1_scooter': 0.8176, 'f1_bike': 0.6750, 'f1_motorcycle': 0.4000,
    'f1_jog': 0.0000, 'f1_train': 0.0000, 'round': 10,
}
ref_df   = pd.DataFrame([fedavg_ref])
all_last = pd.concat([ref_df, last_rows], ignore_index=True)

print(f"\n{'='*65}")
print("FINAL COMPARISON  (Round 10, 60s windows)")
print(f"{'='*65}")

print(f"\n  {'Experiment':<30} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*50}")
for _, row in all_last.iterrows():
    print(f"  {row['experiment']:<30} {float(row['global_acc']):>7.4f}  {float(row['global_macro_f1']):>8.4f}")

print(f"\n  Per-city Macro F1:")
header = f"  {'Experiment':<30} " + "  ".join(f"{c[:6]:>8}" for c in CITIES)
print(header)
print(f"  {'-'*62}")
for _, row in all_last.iterrows():
    vals = "  ".join(f"{float(row[f'{c}_macro_f1']):>8.4f}" for c in CITIES)
    print(f"  {row['experiment']:<30} {vals}")

print(f"\n  Per-class F1:")
header2 = f"  {'Experiment':<30} " + "  ".join(f"{cls[:5]:>7}" for cls in CLASS_NAMES)
print(header2)
print(f"  {'-'*90}")
for _, row in all_last.iterrows():
    vals = "  ".join(f"{float(row[f'f1_{cls}']):>7.3f}" for cls in CLASS_NAMES)
    print(f"  {row['experiment']:<30} {vals}")

print(f"\nAll results saved to: {SAVE_DIR}")
print("Done.")
