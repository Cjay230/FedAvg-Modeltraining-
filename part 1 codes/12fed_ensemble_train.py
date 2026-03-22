import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier
#This file is the second experiment. It keeps the same federated training setup, but modifies the local training data to reduce class imbalance, 
# so I can compare the effect of balancing against the baseline FedAvg model.
#keeps the federated MLP
#adds local Random Forest and Gradient Boosting
#combines them with majority voting
# =========================================================
# PATHS
# =========================================================
DATA_ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients\fl_ready")
RESULTS = Path(r"C:\Users\user\Desktop\VIP\federated\results\fed_ensemble_binary")
RESULTS.mkdir(parents=True, exist_ok=True)

clients = ["jeddah", "kaust", "kz", "mekkah"]

with open(DATA_ROOT / "feature_cols.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

with open(DATA_ROOT / "label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# =========================================================
# SETTINGS
# =========================================================
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
HIDDEN = (256, 256)
LEARNING_RATE_INIT = 0.01
RANDOM_STATE = 42

config = {
    "experiment": "fed_ensemble_binary",
    "clients": clients,
    "task": ["car", "walk"],
    "feature_count": len(feature_cols),
    "feature_cols": feature_cols,
    "label_map": label_map,
    "federated_component": "MLP + FedAvg",
    "local_components": ["RandomForest", "GradientBoosting"],
    "fusion": "majority_vote",
    "rounds": ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "hidden_layers": HIDDEN,
    "learning_rate_init": LEARNING_RATE_INIT,
}

with open(RESULTS / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

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
# FEDERATED MLP
# =========================================================
def create_mlp():
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

global_mlp = create_mlp()
X0, y0, _, _ = client_data[clients[0]]
global_mlp.fit(X0[:1000], y0[:1000])  # initialize architecture

round_rows = []

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== FED ROUND {rnd} ===")

    global_coefs, global_intercepts = get_weights(global_mlp)
    local_params = []
    sample_counts = []

    for city in clients:
        X_train, y_train, X_test, y_test = client_data[city]

        local_mlp = create_mlp()
        local_mlp.fit(X0[:1000], y0[:1000])  # initialize
        set_weights(local_mlp, global_coefs, global_intercepts)

        local_mlp.fit(X_train, y_train)

        local_params.append(get_weights(local_mlp))
        sample_counts.append(len(X_train))

    new_coefs, new_intercepts = fedavg(local_params, sample_counts)
    set_weights(global_mlp, new_coefs, new_intercepts)

    # track federated-MLP-only round metrics
    city_accs = []
    for city in clients:
        X_train, y_train, X_test, y_test = client_data[city]
        preds = global_mlp.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )

        city_accs.append(acc)
        round_rows.append({
            "round": rnd,
            "city": city,
            "model": "fed_mlp",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        print(city, f"fed_mlp acc={acc:.4f} f1={f1:.4f}")

    print("fed_mlp mean accuracy:", round(float(np.mean(city_accs)), 4))

pd.DataFrame(round_rows).to_csv(RESULTS / "round_metrics_fed_mlp.csv", index=False)

# =========================================================
# LOCAL ENSEMBLE COMPONENTS + MAJORITY VOTE
# =========================================================
def majority_vote_binary(pred1, pred2, pred3):
    votes = pred1 + pred2 + pred3
    return (votes >= 2).astype(int)

final_rows = []
all_y_true = []
all_y_pred_ensemble = []

for city in clients:
    print(f"\nTraining local ensemble models for {city} ...")

    X_train, y_train, X_test, y_test = client_data[city]

    # local RF
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # local GB
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)

    # predictions
    pred_mlp = global_mlp.predict(X_test)
    pred_rf = rf.predict(X_test)
    pred_gb = gb.predict(X_test)

    pred_ensemble = majority_vote_binary(pred_mlp, pred_rf, pred_gb)

    # save component metrics too
    for model_name, preds in [
        ("fed_mlp", pred_mlp),
        ("local_rf", pred_rf),
        ("local_gb", pred_gb),
        ("ensemble_vote", pred_ensemble),
    ]:
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])

        final_rows.append({
            "city": city,
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

        # save confusion matrix for each city/model
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{inv_label_map[0]}", f"true_{inv_label_map[1]}"],
            columns=[f"pred_{inv_label_map[0]}", f"pred_{inv_label_map[1]}"],
        )
        cm_df.to_csv(RESULTS / f"confusion_matrix_{city}_{model_name}.csv")

    all_y_true.extend(y_test.tolist())
    all_y_pred_ensemble.extend(pred_ensemble.tolist())

final_df = pd.DataFrame(final_rows)
final_df.to_csv(RESULTS / "final_metrics_per_city_and_model.csv", index=False)

# =========================================================
# GLOBAL ENSEMBLE SUMMARY
# =========================================================
global_acc = accuracy_score(all_y_true, all_y_pred_ensemble)
global_prec, global_rec, global_f1, _ = precision_recall_fscore_support(
    all_y_true, all_y_pred_ensemble, average="binary", zero_division=0
)
global_cm = confusion_matrix(all_y_true, all_y_pred_ensemble, labels=[0, 1])

global_summary = {
    "experiment": "fed_ensemble_binary",
    "global_accuracy": float(global_acc),
    "global_precision": float(global_prec),
    "global_recall": float(global_rec),
    "global_f1": float(global_f1),
    "label_map": label_map,
    "clients": clients,
    "feature_count": len(feature_cols),
    "rounds": ROUNDS,
    "ensemble_members": ["fed_mlp", "local_rf", "local_gb"],
    "fusion": "majority_vote",
}

with open(RESULTS / "global_summary.json", "w", encoding="utf-8") as f:
    json.dump(global_summary, f, indent=2)

pd.DataFrame(
    global_cm,
    index=[f"true_{inv_label_map[0]}", f"true_{inv_label_map[1]}"],
    columns=[f"pred_{inv_label_map[0]}", f"pred_{inv_label_map[1]}"],
).to_csv(RESULTS / "confusion_matrix_global_ensemble.csv")

print("\nSaved results to:")
print(RESULTS)
print("\nFinal metrics preview:")
print(final_df)
print("\nGlobal ensemble summary:")
print(global_summary)