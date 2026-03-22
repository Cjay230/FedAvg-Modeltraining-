import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Once I had the local client models, I wanted to introduce collaboration between them. This script implements a federated-style ensemble. 
# Instead of aggregating weights like FedAvg,it combines the predictions of the local client models using majority voting. 
# So this simulates collaborative inference without sharing raw data
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# =============================
# PATHS
# =============================

clients_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\clients"
save_root = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\federated_ensemble"

os.makedirs(save_root, exist_ok=True)

client_files = sorted([
    f for f in os.listdir(clients_dir)
    if f.endswith(".csv") and f.startswith("client_")
])

# =============================
# HELPERS
# =============================

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

def majority_vote(row):
    counts = Counter(row)
    return counts.most_common(1)[0][0]

# =============================
# BUILD SHARED TEST SET
# =============================

print("Loading all client datasets...")
all_clients = []
for file in client_files:
    df = pd.read_csv(os.path.join(clients_dir, file), low_memory=False)
    all_clients.append(df)

full_df = pd.concat(all_clients, ignore_index=True)

full_df = full_df.dropna(subset=["label"]).copy()

for col in full_df.columns:
    if col != "label":
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

full_df = full_df.fillna(0)

X = full_df.drop(columns=["label"]).copy()
y = full_df["label"].astype(str).copy()

# shared test set
_, X_test_shared, _, y_test_shared = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Shared test set size:", len(X_test_shared))

# =============================
# TRAIN ONE MODEL PER CLIENT
# =============================

all_predictions = {}
client_metrics = []

for file in client_files:
    client_name = os.path.splitext(file)[0]
    print(f"\nTraining local model for {client_name}...")

    df = pd.read_csv(os.path.join(clients_dir, file), low_memory=False)
    df = df.dropna(subset=["label"]).copy()

    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    X_client = df.drop(columns=["label"]).copy()
    y_client = df["label"].astype(str).copy()

    X_train_client, _, y_train_client, _ = train_test_split(
        X_client,
        y_client,
        test_size=0.2,
        random_state=42,
        stratify=y_client
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train_client, y_train_client)

    y_pred_client = model.predict(X_test_shared)
    all_predictions[client_name] = y_pred_client

    metrics = compute_metrics(y_test_shared, y_pred_client)
    metrics["client"] = client_name
    client_metrics.append(metrics)

    client_dir = os.path.join(save_root, client_name)
    os.makedirs(client_dir, exist_ok=True)

    pd.DataFrame({
        "y_true": y_test_shared.values,
        "y_pred": y_pred_client
    }).to_csv(os.path.join(client_dir, "shared_test_predictions.csv"), index=False)

    save_json(metrics, os.path.join(client_dir, "shared_test_metrics.json"))

# =============================
# MAJORITY VOTING ENSEMBLE
# =============================

pred_df = pd.DataFrame(all_predictions)
ensemble_pred = pred_df.apply(majority_vote, axis=1)

ensemble_metrics = compute_metrics(y_test_shared, ensemble_pred)

pd.DataFrame({
    "y_true": y_test_shared.values,
    "y_pred": ensemble_pred
}).to_csv(os.path.join(save_root, "ensemble_predictions.csv"), index=False)

save_json(ensemble_metrics, os.path.join(save_root, "ensemble_metrics.json"))

# =============================
# SAVE SUMMARY
# =============================

client_metrics_df = pd.DataFrame(client_metrics)
client_metrics_df.to_csv(os.path.join(save_root, "client_shared_test_metrics_summary.csv"), index=False)

comparison_df = pd.DataFrame(client_metrics + [{
    "client": "ensemble_majority_vote",
    **ensemble_metrics
}])
comparison_df.to_csv(os.path.join(save_root, "comparison_with_ensemble.csv"), index=False)

# =============================
# CONFUSION MATRIX FOR ENSEMBLE
# =============================

labels = sorted(y.unique())
cm = confusion_matrix(y_test_shared, ensemble_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(os.path.join(save_root, "ensemble_confusion_matrix.csv"))

report = classification_report(
    y_test_shared, ensemble_pred, labels=labels, output_dict=True, zero_division=0
)
save_json(report, os.path.join(save_root, "ensemble_classification_report.json"))

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Federated-Style Ensemble Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha="right")
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(save_root, "ensemble_confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plot_df = comparison_df.copy()
plt.bar(plot_df["client"], plot_df["macro_f1"])
plt.title("Client Models vs Ensemble - Macro F1")
plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_root, "client_vs_ensemble_macro_f1.png"), dpi=200, bbox_inches="tight")
plt.close()

print("\nEnsemble metrics:")
print(ensemble_metrics)

print("\nSaved:")
print(os.path.join(save_root, "comparison_with_ensemble.csv"))
print(os.path.join(save_root, "ensemble_metrics.json"))
print(os.path.join(save_root, "ensemble_confusion_matrix.png"))
print(os.path.join(save_root, "client_vs_ensemble_macro_f1.png"))