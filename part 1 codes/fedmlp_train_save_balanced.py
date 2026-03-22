import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier

DATA_ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients\fl_ready")
RESULTS = Path(r"C:\Users\user\Desktop\VIP\federated\results\fedmlp_binary_balanced")
RESULTS.mkdir(parents=True, exist_ok=True)

clients = ["jeddah", "kaust", "kz", "mekkah"]

with open(DATA_ROOT / "feature_cols.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

with open(DATA_ROOT / "label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
HIDDEN = (256, 256)
LEARNING_RATE_INIT = 0.01
RANDOM_STATE = 42

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
    "balancing": "downsample majority class per client train set",
}

with open(RESULTS / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="sgd",
        learning_rate_init=LEARNING_RATE_INIT,
        batch_size=BATCH_SIZE,
        max_iter=LOCAL_EPOCHS,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

def get_weights(model):
    return [w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_]

def set_weights(model, coefs, intercepts):
    model.coefs_ = [w.copy() for w in coefs]
    model.intercepts_ = [b.copy() for b in intercepts]

def fedavg(local_params, sample_counts):
    total = np.sum(sample_counts)

    avg_coefs = []
    avg_intercepts = []

    for layer in range(len(local_params[0][0])):
        layer_sum = None
        for (coefs, _), n in zip(local_params, sample_counts):
            weighted = coefs[layer] * (n / total)
            layer_sum = weighted if layer_sum is None else layer_sum + weighted
        avg_coefs.append(layer_sum)

    for layer in range(len(local_params[0][1])):
        layer_sum = None
        for (_, intercepts), n in zip(local_params, sample_counts):
            weighted = intercepts[layer] * (n / total)
            layer_sum = weighted if layer_sum is None else layer_sum + weighted
        avg_intercepts.append(layer_sum)

    return avg_coefs, avg_intercepts

def balance_train_df(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["y"].value_counts()
    if len(counts) < 2:
        return df.copy()

    min_count = counts.min()
    balanced_parts = []

    for cls in sorted(counts.index):
        part = df[df["y"] == cls].sample(
            n=min_count,
            random_state=RANDOM_STATE,
            replace=False
        )
        balanced_parts.append(part)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced

client_data = {}
balance_rows = []

for city in clients:
    train = pd.read_csv(DATA_ROOT / f"{city}_train.csv")
    test = pd.read_csv(DATA_ROOT / f"{city}_test.csv")

    original_counts = train["mode"].value_counts().to_dict()
    balanced_train = balance_train_df(train)
    balanced_counts = balanced_train["mode"].value_counts().to_dict()

    X_train = balanced_train[feature_cols].to_numpy(dtype=np.float64)
    y_train = balanced_train["y"].to_numpy(dtype=np.int64)

    X_test = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test["y"].to_numpy(dtype=np.int64)

    client_data[city] = (X_train, y_train, X_test, y_test)

    balance_rows.append({
        "city": city,
        "original_train_rows": int(len(train)),
        "balanced_train_rows": int(len(balanced_train)),
        "original_train_car": int(original_counts.get("car", 0)),
        "original_train_walk": int(original_counts.get("walk", 0)),
        "balanced_train_car": int(balanced_counts.get("car", 0)),
        "balanced_train_walk": int(balanced_counts.get("walk", 0)),
    })

balance_df = pd.DataFrame(balance_rows)
balance_df.to_csv(RESULTS / "balance_summary.csv", index=False)

print("Loaded balanced clients:")
for city in clients:
    X_train, y_train, X_test, y_test = client_data[city]
    print(city, "train:", X_train.shape, "test:", X_test.shape)

global_model = create_model()
X0, y0, _, _ = client_data[clients[0]]
global_model.fit(X0[:1000], y0[:1000])

round_rows = []

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== ROUND {rnd} ===")

    global_coefs, global_intercepts = get_weights(global_model)
    local_params = []
    sample_counts = []

    for city in clients:
        X_train, y_train, X_test, y_test = client_data[city]

        local_model = create_model()
        local_model.fit(X0[:1000], y0[:1000])
        set_weights(local_model, global_coefs, global_intercepts)
        local_model.fit(X_train, y_train)

        local_params.append(get_weights(local_model))
        sample_counts.append(len(X_train))

    new_coefs, new_intercepts = fedavg(local_params, sample_counts)
    set_weights(global_model, new_coefs, new_intercepts)

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

    print("mean accuracy:", round(float(np.mean(city_accs)), 4))

pd.DataFrame(round_rows).to_csv(RESULTS / "round_metrics.csv", index=False)

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
    "balancing": "downsample majority class per client train set",
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