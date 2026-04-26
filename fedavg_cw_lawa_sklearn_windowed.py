# cwFedAvg + LAWA (gamma=0.2) on windowed data using sklearn MLPClassifier
import os, copy, time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

WINDOWED_DIR = r"C:\Users\user\Desktop\electives\VIP\kaust data\windowed"
RESULTS_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3,
               'bike': 4, 'motorcycle': 5, 'jog': 6}
CLASS_NAMES = ['car', 'walk', 'bus', 'scooter', 'bike', 'motorcycle', 'jog']
N_CLASSES   = 7
N_ROUNDS    = 10
GAMMA       = 0.2
LAWA_MAX    = 50_000
META_COLS   = ['label', 'source_file', 'window_id', 't_start_sec', 't_end_sec']
ALL_CLASSES = list(range(N_CLASSES))

np.random.seed(42)

# ── Load & split ─────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING & SPLITTING DATA")
print("=" * 60)

sample_df  = pd.read_csv(f'{WINDOWED_DIR}/jeddah_windowed.csv')
feat_cols  = [c for c in sample_df.columns if c not in META_COLS]
N_FEATURES = len(feat_cols)

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
city_classes = {}

for city in CITIES:
    df = pd.read_csv(f'{WINDOWED_DIR}/{city}_windowed.csv')
    df['label_enc'] = df['label'].map(LABEL_MAP)
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values.astype(np.int32)
    city_classes[city] = set(y.tolist())
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_d[city], X_test_d[city] = Xtr, Xte
    y_train_d[city], y_test_d[city] = ytr, yte
    present = [CLASS_NAMES[c] for c in sorted(city_classes[city])]
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  classes={present}")

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c] for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test set: {len(y_test_global)} samples")

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_weights(mlp):
    return [w.copy() for w in mlp.coefs_], [b.copy() for b in mlp.intercepts_]

def set_weights(mlp, coefs, intercepts):
    mlp.coefs_      = [w.copy() for w in coefs]
    mlp.intercepts_ = [b.copy() for b in intercepts]

def make_template():
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        solver='sgd',
        learning_rate_init=0.01,
        momentum=0.9,
        batch_size=32,
        max_iter=1,
        random_state=42
    )
    X_seed = np.zeros((N_CLASSES, N_FEATURES), dtype=np.float32)
    y_seed = np.arange(N_CLASSES, dtype=np.int32)
    mlp.partial_fit(X_seed, y_seed, classes=ALL_CLASSES)
    return mlp

def class_count_vec(y):
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        v[c] = (y == c).sum()
    return v

def train_client(template, g_coefs, g_biases, city):
    """Local training with knowledge inheritance for missing classes."""
    local = copy.deepcopy(template)
    set_weights(local, g_coefs, g_biases)
    local.partial_fit(X_train_d[city], y_train_d[city])

    # Knowledge inheritance: restore global output weights for missing classes
    missing = [c for c in ALL_CLASSES if c not in city_classes[city]]
    if missing:
        for c in missing:
            local.coefs_[-1][:, c]  = g_coefs[-1][:, c]
            local.intercepts_[-1][c] = g_biases[-1][c]
    return local

def lawa_weights(local_models):
    """Compute normalised LAWA scalar weight per client (L_min/L_max)."""
    n     = len(local_models)
    raw_w = np.zeros(n)
    for k, mlp in enumerate(local_models):
        X, y = X_train_d[CITIES[k]], y_train_d[CITIES[k]]
        if len(X) > LAWA_MAX:
            idx  = np.random.choice(len(X), LAWA_MAX, replace=False)
            X, y = X[idx], y[idx]
        log_p  = mlp.predict_log_proba(X)
        losses = []
        for c in ALL_CLASSES:
            mask = (y == c)
            if mask.sum() > 0:
                losses.append(float(-log_p[mask, c].mean()))
        if len(losses) < 2:
            raw_w[k] = 1.0
        else:
            L_min, L_max = min(losses), max(losses)
            raw_w[k] = 1.0 if L_max == 0 else L_min / L_max
    s = raw_w.sum()
    return raw_w / s if s > 0 else np.ones(n) / n

def agg_blended(local_models, gamma):
    """cwFedAvg + LAWA blend aggregation."""
    n      = len(local_models)
    cc     = np.array([class_count_vec(y_train_d[c]) for c in CITIES])  # (n, 7)
    sizes  = cc.sum(axis=1)
    total  = sizes.sum()
    w_scalar = sizes / total                                             # (n,) hidden layers

    ct   = cc.sum(axis=0)                                                # (7,)
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)       # (n, 7) output layer

    lawa_w = lawa_weights(local_models)                                  # (n,) normalised

    # Blended scalar (hidden layers)
    bl_s = gamma * lawa_w + (1 - gamma) * w_scalar
    bl_s /= bl_s.sum()

    # Blended class-wise (output layer)
    bl_cw = gamma * lawa_w[:, None] + (1 - gamma) * w_cw               # (n, 7)
    for c in range(N_CLASSES):
        s = bl_cw[:, c].sum()
        if s > 0:
            bl_cw[:, c] /= s

    n_layers = len(local_models[0].coefs_)
    new_coefs, new_biases = [], []

    for l in range(n_layers):
        is_out = (l == n_layers - 1)
        if is_out:
            # Output weight matrix: (hidden, N_CLASSES)
            W = np.zeros_like(local_models[0].coefs_[l])
            B = np.zeros_like(local_models[0].intercepts_[l])
            for c in range(N_CLASSES):
                for k in range(n):
                    W[:, c] += bl_cw[k, c] * local_models[k].coefs_[l][:, c]
                    B[c]    += bl_cw[k, c] * local_models[k].intercepts_[l][c]
            new_coefs.append(W)
            new_biases.append(B)
        else:
            new_coefs.append(sum(bl_s[k] * local_models[k].coefs_[l]      for k in range(n)))
            new_biases.append(sum(bl_s[k] * local_models[k].intercepts_[l] for k in range(n)))

    return new_coefs, new_biases

def evaluate(template):
    y_pred = template.predict(X_test_global)
    g_acc  = float(accuracy_score(y_test_global, y_pred))
    g_f1   = float(f1_score(y_test_global, y_pred, average='macro',
                             labels=ALL_CLASSES, zero_division=0))
    pc_f1  = f1_score(y_test_global, y_pred, average=None,
                      labels=ALL_CLASSES, zero_division=0)
    city_m = {}
    for city in CITIES:
        yp = template.predict(X_test_d[city])
        city_m[city] = {
            'acc': float(accuracy_score(y_test_d[city], yp)),
            'f1':  float(f1_score(y_test_d[city], yp, average='macro',
                                  labels=ALL_CLASSES, zero_division=0))
        }
    return g_acc, g_f1, pc_f1, city_m

# ── Run ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"cwFedAvg + LAWA (gamma={GAMMA}) — sklearn MLP — windowed data")
print(f"{'='*60}")

template = make_template()
g_coefs, g_biases = get_weights(template)
rows = []

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    local_models = [train_client(template, g_coefs, g_biases, city) for city in CITIES]
    g_coefs, g_biases = agg_blended(local_models, GAMMA)
    set_weights(template, g_coefs, g_biases)

    g_acc, g_f1, pc_f1, city_m = evaluate(template)
    print(f"  Round {rnd:2d}/10  GlobalAcc {g_acc:.4f}  GlobalMacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'round': rnd, 'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
    for city in CITIES:
        row[f'{city}_acc']      = round(city_m[city]['acc'], 6)
        row[f'{city}_macro_f1'] = round(city_m[city]['f1'],  6)
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(RESULTS_DIR, 'cwfedavg_lawa_sklearn_gamma02_windowed.csv')
df.to_csv(out, index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
last = df[df['round'] == N_ROUNDS].iloc[0]
print(f"\n--- Round 10 Results ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

print(f"\n--- Comparison vs other experiments ---")
comparison = [
    ("sklearn FedAvg (raw rows)",          0.4139, 0.2869),
    ("sklearn FedAvg (windowed)",          0.8037, 0.6333),
    ("PyTorch FedAvg (windowed)",          0.7336, 0.4935),
    ("PyTorch cwFedAvg+LAWA g=0.2",        0.7664, 0.6443),
    (f"sklearn cwFedAvg+LAWA g={GAMMA} (windowed)", last['global_acc'], last['global_macro_f1']),
]
print(f"  {'Experiment':<42} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*62}")
for name, acc, f1 in comparison:
    print(f"  {name:<42} {acc:>7.4f}  {f1:>8.4f}")

print(f"\nSaved -> {out}")
