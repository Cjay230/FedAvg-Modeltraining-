import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier
#his is the baseline federated learning implementation. It creates the global MLP model, sends it to each city, each client trains locally, 
# and then the server aggregates the local weights using Federated Averaging. This loop is repeated for the communication rounds.
# =========================================================
# PATHS
# =========================================================
DATA_ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients\fl_ready")
RESULTS = Path(r"C:\Users\user\Desktop\VIP\federated\results\fedmlp_binary")
RESULTS.mkdir(parents=True, exist_ok=True)

clients = ["jeddah", "kaust", "kz", "mekkah"]

with open(DATA_ROOT / "feature_cols.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

with open(DATA_ROOT / "label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# =========================================================
# LOAD DATA
# =========================================================
client_data = {}

for city in clients:
    train = pd.read_csv(DATA_ROOT / f"{city}_train.csv")
    test = pd.read_csv(DATA_ROOT / f"{city}_test.csv")

    X_train = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["y"].to_numpy(dtype=np.int64)

    X_test = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test["y"].to_numpy(dtype=np.int64)

    client_data[city] = (X_train, y_train, X_test, y_test)

print("Loaded clients:")
for city in clients:
    X_train, y_train, X_test, y_test = client_data[city]
    print(city, "train:", X_train.shape, "test:", X_test.shape)

# =========================================================
# MODEL SETTINGS
# Paper-inspired:
# - MLP
# - SGD
# - batch size 32
# - hidden layers 256,256
# =========================================================
ROUNDS = 10          # start smaller for stability; can increase later
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
HIDDEN = (256, 256)
LEARNING_RATE_INIT = 0.01

config = {
    "rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "hidden_layers": HIDDEN,
    "learning_rate_init": LEARNING_RATE_INIT,
    "solver": "sgd",
    "shared_task": ["car", "walk"],
    "clients": clients,
    "feature_count": len(feature_cols),
    "label_map": label_map,
}

with open(RESULTS / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# =========================================================
# MODEL CREATION
# =========================================================
def create_model():
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="sgd",
        learning_rate_init=LEARNING_RATE_INIT,
        batch_size=BATCH_SIZE,
        max_iter=LOCAL_EPOCHS,
        warm_start=True,
        random_state=42,
    )

# initialize on first client's data so sklearn creates the weight shapes
global_model = create_model()
X0, y0, _, _ = client_data[clients[0]]
global_model.fit(X0[:1000], y0[:1000])  # tiny init subset is enough

# =========================================================
# FEDAVG HELPERS
# sklearn MLP has:
# - coefs_
# - intercepts_
# =========================================================
def get_weights(model):
    return [w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_]

def set_weights(model, coefs, intercepts):
    model.coefs_ = [w.copy() for w in coefs]
    model.intercepts_ = [b.copy() for b in intercepts]

def fedavg(local_params, sample_counts):
    total = np.sum(sample_counts)

    avg_coefs = []
    avg_intercepts = []

    n_layers = len(local_params[0][0])

    for layer in range(n_layers):
        layer_sum = None
        for (coefs, _), n in zip(local_params, sample_counts):
            weighted = coefs[layer] * (n / total)
            layer_sum = weighted if layer_sum is None else layer_sum + weighted
        avg_coefs.append(layer_sum)

    n_bias_layers = len(local_params[0][1])

    for layer in range(n_bias_layers):
        layer_sum = None
        for (_, intercepts), n in zip(local_params, sample_counts):
            weighted = intercepts[layer] * (n / total)
            layer_sum = weighted if layer_sum is None else layer_sum + weighted
        avg_intercepts.append(layer_sum)

    return avg_coefs, avg_intercepts

# =========================================================
# ROUND METRICS
# =========================================================
round_rows = []

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== ROUND {rnd} ===")

    global_coefs, global_intercepts = get_weights(global_model)

    local_params = []
    sample_counts = []

    # local training
    for city in clients:
        X_train, y_train, X_test, y_test = client_data[city]

        local_model = create_model()
        local_model.fit(X0[:1000], y0[:1000])  # initialize structure
        set_weights(local_model, global_coefs, global_intercepts)

        local_model.fit(X_train, y_train)

        local_params.append(get_weights(local_model))
        sample_counts.append(len(X_train))

    # aggregate
    new_coefs, new_intercepts = fedavg(local_params, sample_counts)
    set_weights(global_model, new_coefs, new_intercepts)

    # evaluate per city
    city_accs = []

    for city in clients:
        X_train, y_train, X_test, y_test = client_data[city]

        preds = global_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )

        city_accs.append(acc)

        round_rows.append({
            "round": rnd,
            "city": city,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        print(city, f"acc={acc:.4f} f1={f1:.4f}")

    mean_acc = float(np.mean(city_accs))
    print("mean accuracy:", round(mean_acc, 4))

# save per-round metrics
round_df = pd.DataFrame(round_rows)
round_df.to_csv(RESULTS / "round_metrics.csv", index=False)

# =========================================================
# FINAL EVALUATION
# =========================================================
final_rows = []
all_y_true = []
all_y_pred = []

for city in clients:
    X_train, y_train, X_test, y_test = client_data[city]

    preds = global_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    final_rows.append({
        "city": city,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    })

    # save city confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{inv_label_map[0]}", f"true_{inv_label_map[1]}"],
        columns=[f"pred_{inv_label_map[0]}", f"pred_{inv_label_map[1]}"],
    )
    cm_df.to_csv(RESULTS / f"confusion_matrix_{city}.csv")

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(preds.tolist())

final_df = pd.DataFrame(final_rows)
final_df.to_csv(RESULTS / "final_metrics_per_city.csv", index=False)

# global overall metrics
global_acc = accuracy_score(all_y_true, all_y_pred)
global_prec, global_rec, global_f1, _ = precision_recall_fscore_support(
    all_y_true, all_y_pred, average="binary", zero_division=0
)
global_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])

global_summary = {
    "global_accuracy": float(global_acc),
    "global_precision": float(global_prec),
    "global_recall": float(global_rec),
    "global_f1": float(global_f1),
    "label_map": label_map,
    "clients": clients,
    "feature_count": len(feature_cols),
    "rounds": ROUNDS,
}

with open(RESULTS / "global_summary.json", "w", encoding="utf-8") as f:
    json.dump(global_summary, f, indent=2)

pd.DataFrame(
    global_cm,
    index=[f"true_{inv_label_map[0]}", f"true_{inv_label_map[1]}"],
    columns=[f"pred_{inv_label_map[0]}", f"pred_{inv_label_map[1]}"],
).to_csv(RESULTS / "confusion_matrix_global.csv")

print("\nSaved results to:")
print(RESULTS)
print("\nFinal per-city metrics:")
print(final_df)
print("\nGlobal summary:")
print(global_summary)