# PyTorch FedAvg across window sizes: 60s, 120s, 180s, 300s
# For each window size: window raw data -> train FedAvg -> report results
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
from scipy import stats as sp_stats

RAW_DIR   = r"C:\Users\user\Desktop\electives\VIP\kaust data\full data"
WIN_DIR   = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(WIN_DIR,  exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
MIN_ROWS    = 10
WINDOW_SIZES = [60, 120, 180, 300]

torch.manual_seed(42)
np.random.seed(42)

ALL_CLASSES = list(range(N_CLASSES))

# -- Column identification -----------------------------------------------------
def find_cols(df, keywords):
    return [c for c in df.columns if any(k.lower() in c.lower() for k in keywords)]

# -- Window feature computation ------------------------------------------------
def window_features(win, rsrp_cols, rssi_cols, rsrq_cols, cqi0_cols, cqi1_cols,
                    rank_cols, vel_cols, lon_col, lat_col,
                    ta_col, cell_id_col, pdsch_col, pathloss_col, window_sec):
    feats = {'n_rows': len(win)}
    for gname, cols in [('rsrp',rsrp_cols),('rssi',rssi_cols),('rsrq',rsrq_cols)]:
        vals = win[cols].apply(pd.to_numeric, errors='coerce').values.flatten() if cols else np.array([])
        vals = vals[~np.isnan(vals)]
        feats[f'{gname}_mean'] = float(np.mean(vals))  if len(vals)>0 else np.nan
        feats[f'{gname}_var']  = float(np.var(vals))   if len(vals)>1 else np.nan
        feats[f'{gname}_max']  = float(np.max(vals))   if len(vals)>0 else np.nan
        feats[f'{gname}_min']  = float(np.min(vals))   if len(vals)>0 else np.nan
    for gname, cols in [('cqi0',cqi0_cols),('cqi1',cqi1_cols)]:
        vals = win[cols].apply(pd.to_numeric, errors='coerce').values.flatten() if cols else np.array([])
        vals = vals[~np.isnan(vals)]
        feats[f'{gname}_mean'] = float(np.mean(vals)) if len(vals)>0 else np.nan
        feats[f'{gname}_var']  = float(np.var(vals))  if len(vals)>1 else np.nan
    if rank_cols:
        vals = win[rank_cols].apply(pd.to_numeric, errors='coerce').values.flatten()
        vals = vals[~np.isnan(vals)]
        feats['rank_mean'] = float(np.mean(vals)) if len(vals)>0 else np.nan
    else:
        feats['rank_mean'] = np.nan
    for col, key in [(vel_cols[0] if vel_cols else None, 'velocity'),
                     (lon_col, 'longitude'), (lat_col, 'latitude'),
                     (ta_col, 'timing_advance'), (pdsch_col, 'pdsch'),
                     (pathloss_col, 'pathloss')]:
        if col:
            v = win[col].apply(pd.to_numeric, errors='coerce').dropna()
            if key in ('velocity', 'pathloss'):
                feats[f'{key}_mean'] = float(v.mean()) if len(v)>0 else np.nan
                feats[f'{key}_max']  = float(v.max())  if len(v)>0 else np.nan
                feats[f'{key}_var']  = float(v.var())  if len(v)>1 else np.nan
            else:
                feats[f'{key}_mean'] = float(v.mean()) if len(v)>0 else np.nan
        else:
            if key == 'velocity':
                feats['velocity_mean'] = feats['velocity_max'] = feats['velocity_var'] = np.nan
            elif key == 'pathloss':
                feats['pathloss_mean'] = feats['pathloss_max'] = feats['pathloss_var'] = np.nan
            else:
                feats[f'{key}_mean'] = np.nan
    if cell_id_col:
        cell_ids   = win[cell_id_col].apply(pd.to_numeric, errors='coerce').dropna()
        transitions = (cell_ids.diff() != 0).sum()
        feats['handover_rate'] = float(transitions / window_sec)
        feats['unique_cells']  = int(cell_ids.nunique())
    else:
        feats['handover_rate'] = feats['unique_cells'] = np.nan
    return feats

# -- Windowing for one city ----------------------------------------------------
def build_windows(city, window_sec):
    cache = f'{WIN_DIR}/{city}_{window_sec}s.csv'
    if os.path.exists(cache):
        return pd.read_csv(cache)

    df = pd.read_csv(f'{RAW_DIR}/{city}.csv', low_memory=False)
    df['time_parsed'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
    df = df.sort_values(['source_file','time_parsed']).reset_index(drop=True)

    rsrp_cols    = find_cols(df, ['RSRP/antenna'])
    rssi_cols    = find_cols(df, ['E-UTRAN carrier RSSI'])
    rsrq_cols    = find_cols(df, ['RSRQ/antenna'])
    cqi0_cols    = find_cols(df, ['CQI for codeword 0'])
    cqi1_cols    = find_cols(df, ['CQI for codeword 1'])
    rank_cols    = find_cols(df, ['Rank indication -'])
    vel_cols     = find_cols(df, ['Velocity'])
    lon_col      = next((c for c in df.columns if 'longitude' in c.lower()), None)
    lat_col      = next((c for c in df.columns if 'latitude'  in c.lower()), None)
    ta_col       = next((c for c in df.columns if 'timing advance' in c.lower()), None)
    cell_id_col  = next((c for c in df.columns if 'physical cell identity (lte pcell)' in c.lower()), None)
    pdsch_col    = next((c for c in df.columns if 'pdsch scheduled' in c.lower()), None)
    pathloss_col = next((c for c in df.columns if 'pathloss' in c.lower()), None)

    windows = []
    for src, grp in df.groupby('source_file'):
        grp = grp.dropna(subset=['time_parsed']).copy()
        if len(grp) < MIN_ROWS:
            continue
        grp['t_sec']    = (grp['time_parsed'] - grp['time_parsed'].min()).dt.total_seconds()
        grp['win_id']   = (grp['t_sec'] // window_sec).astype(int)
        for wid, win in grp.groupby('win_id'):
            if len(win) < MIN_ROWS:
                continue
            feats = window_features(win, rsrp_cols, rssi_cols, rsrq_cols,
                                    cqi0_cols, cqi1_cols, rank_cols,
                                    vel_cols, lon_col, lat_col,
                                    ta_col, cell_id_col, pdsch_col, pathloss_col,
                                    window_sec)
            lbl = win['label'].mode()
            feats['label'] = lbl.iloc[0] if len(lbl)>0 else 'unknown'
            windows.append(feats)

    result = pd.DataFrame(windows)
    result.to_csv(cache, index=False)
    return result

# -- Model ---------------------------------------------------------------------
def make_mlp(n_features):
    return nn.Sequential(
        nn.Linear(n_features, 256), nn.ReLU(),
        nn.Linear(256, 256),        nn.ReLU(),
        nn.Linear(256, N_CLASSES)
    )

# -- FedAvg training -----------------------------------------------------------
def run_fedavg(window_sec, feat_cols, X_train_d, X_test_d, y_train_d, y_test_d,
               X_test_global, y_test_global):

    N_FEAT = len(feat_cols)

    def get_params(model):
        return copy.deepcopy(model.state_dict())

    def set_params(model, state):
        model.load_state_dict(copy.deepcopy(state))

    def train_local(global_state):
        results = []
        for city in CITIES:
            m = make_mlp(N_FEAT)
            set_params(m, global_state)
            m.train()
            opt  = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
            loss_fn = nn.CrossEntropyLoss()
            loader = DataLoader(
                TensorDataset(torch.tensor(X_train_d[city]),
                              torch.tensor(y_train_d[city])),
                batch_size=BATCH_SIZE, shuffle=True)
            for Xb, yb in loader:
                opt.zero_grad()
                loss_fn(m(Xb), yb).backward()
                opt.step()
            results.append(m)
        return results

    def fedavg_agg(local_models):
        sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
        w     = sizes / sizes.sum()
        new_state = {}
        for key in local_models[0].state_dict():
            new_state[key] = sum(
                w[k] * local_models[k].state_dict()[key].float()
                for k in range(len(CITIES))
            )
        return new_state

    def evaluate(state):
        m = make_mlp(N_FEAT)
        set_params(m, state)
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

    global_model = make_mlp(N_FEAT)
    global_state = get_params(global_model)
    rows = []

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        local_models = train_local(global_state)
        global_state = fedavg_agg(local_models)
        g_acc, g_f1, pc_f1, city_m = evaluate(global_state)
        print(f"    Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

        row = {'window_sec': window_sec, 'round': rnd,
               'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
        for city in CITIES:
            row[f'{city}_macro_f1'] = round(city_m[city], 6)
        for i, cls in enumerate(CLASS_NAMES):
            row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
        rows.append(row)

    return pd.DataFrame(rows)

# -- Main loop -----------------------------------------------------------------
print("=" * 65)
print("PyTorch FedAvg — Window Size Comparison")
print(f"Window sizes: {WINDOW_SIZES}s   |   Rounds: {N_ROUNDS}")
print("=" * 65)

all_results = []
summary_rows = []

for window_sec in WINDOW_SIZES:
    print(f"\n{'-'*65}")
    print(f"  WINDOW SIZE: {window_sec}s")
    print(f"{'-'*65}")

    # Step 1: build windows for each city
    print(f"  Building {window_sec}s windows ...", flush=True)
    city_dfs = {}
    total_windows = 0
    for city in CITIES:
        t0 = time.time()
        df = build_windows(city, window_sec)
        city_dfs[city] = df
        total_windows += len(df)
        dist = df['label'].value_counts().sort_index().to_dict()
        print(f"    {city:8s}: {len(df):4d} windows  dist={dist}  ({time.time()-t0:.1f}s)")
    print(f"  Total windows: {total_windows}")

    # Step 2: determine features
    sample = city_dfs['jeddah']
    feat_cols = [c for c in sample.columns if c != 'label']
    N_FEAT = len(feat_cols)

    # Collect all labels present
    all_labels_present = set()
    for df in city_dfs.values():
        all_labels_present.update(df['label'].unique())
    print(f"  Classes present: {sorted(all_labels_present)}")

    # Step 3: encode labels and split
    X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
    city_sample_counts = {}

    for city in CITIES:
        df = city_dfs[city].copy()
        df['label_enc'] = df['label'].map(LABEL_MAP).fillna(-1).astype(np.int64)
        df = df[df['label_enc'] >= 0]  # drop unknown labels
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        y = df['label_enc'].values
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42)
        X_train_d[city], X_test_d[city] = Xtr, Xte
        y_train_d[city], y_test_d[city] = ytr.astype(np.int64), yte.astype(np.int64)
        city_sample_counts[city] = len(ytr)

    # Step 4: scale
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
    for city in CITIES:
        X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
        X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)
    X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
    y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
    print(f"  Train sizes: { {c: city_sample_counts[c] for c in CITIES} }")
    print(f"  Global test: {len(y_test_global)} samples  |  Features: {N_FEAT}")

    # Step 5: run FedAvg
    print(f"  Training FedAvg ({N_ROUNDS} rounds) ...")
    t0_exp = time.time()
    df_results = run_fedavg(window_sec, feat_cols,
                            X_train_d, X_test_d, y_train_d, y_test_d,
                            X_test_global, y_test_global)
    exp_min = (time.time() - t0_exp) / 60

    # Save per-experiment CSV
    out = os.path.join(SAVE_DIR, f'fedavg_pytorch_{window_sec}s.csv')
    df_results.to_csv(out, index=False)
    all_results.append(df_results)

    last = df_results[df_results['round'] == N_ROUNDS].iloc[0]
    print(f"\n  --- {window_sec}s Window — Round 10 ---")
    print(f"  Global Accuracy : {last['global_acc']:.4f}")
    print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
    print(f"  Per-city Macro F1:")
    for city in CITIES:
        print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")
    print(f"  Time: {exp_min:.1f} min  |  Saved -> {out}")

    summary_rows.append({
        'window_sec':      window_sec,
        'total_windows':   total_windows,
        'global_acc':      round(float(last['global_acc']), 4),
        'global_macro_f1': round(float(last['global_macro_f1']), 4),
        **{f'{city}_macro_f1': round(float(last[f'{city}_macro_f1']), 4) for city in CITIES},
        **{f'f1_{cls}': round(float(last[f'f1_{cls}']), 4) for cls in CLASS_NAMES},
    })

# -- Save combined results -----------------------------------------------------
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(SAVE_DIR, 'fedavg_pytorch_all_windows.csv'), index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(SAVE_DIR, 'fedavg_pytorch_window_summary.csv'), index=False)

# -- Final summary table -------------------------------------------------------
print(f"\n{'='*65}")
print("FINAL SUMMARY — Round 10 across all window sizes")
print(f"{'='*65}")
print(f"\n{'Window':>8}  {'Windows':>8}  {'Acc':>7}  {'MacroF1':>8}")
print(f"{'-'*40}")
for row in summary_rows:
    print(f"  {row['window_sec']:>4}s    {row['total_windows']:>6}    "
          f"{row['global_acc']:>7.4f}  {row['global_macro_f1']:>8.4f}")

print(f"\nAll results saved to: {SAVE_DIR}")
print("Done.")
