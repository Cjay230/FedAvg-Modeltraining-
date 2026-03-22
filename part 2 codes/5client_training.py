import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#After creating the client datasets, this script trains one local Random Forest model per client. Each model only sees its own local subset of the data. 
# The purpose is to measure how much performance changes when the same learning algorithm is trained locally instead of centrally
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

clients_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\clients"
save_root = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\client_models"

os.makedirs(save_root, exist_ok=True)

client_files = [
    f for f in os.listdir(clients_dir)
    if f.endswith(".csv") and f.startswith("client_")
]
client_files.sort()

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

all_metrics = []

for client_file in client_files:
    client_name = os.path.splitext(client_file)[0]
    print(f"\nProcessing {client_name}...")

    client_path = os.path.join(clients_dir, client_file)
    client_save_dir = os.path.join(save_root, client_name)
    os.makedirs(client_save_dir, exist_ok=True)

    df = pd.read_csv(client_path, low_memory=False)

    if "label" not in df.columns:
        raise ValueError(f"{client_file} does not contain 'label' column.")

    df = df.dropna(subset=["label"]).copy()

    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    X = df.drop(columns=["label"]).copy()
    y = df["label"].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["client"] = client_name
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))

    save_json(metrics, os.path.join(client_save_dir, "metrics.json"))

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })
    pred_df.to_csv(os.path.join(client_save_dir, "test_predictions.csv"), index=False)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(client_save_dir, "confusion_matrix.csv"))

    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    save_json(report, os.path.join(client_save_dir, "classification_report.json"))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{client_name} RF Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(client_save_dir, "confusion_matrix.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(metrics)
    all_metrics.append(metrics)

summary_df = pd.DataFrame(all_metrics)
summary_path = os.path.join(save_root, "client_rf_metrics_summary.csv")
summary_df.to_csv(summary_path, index=False)

plt.figure(figsize=(8, 5))
plt.bar(summary_df["client"], summary_df["macro_f1"])
plt.title("Client RF Comparison - Macro F1")
plt.xlabel("Client")
plt.ylabel("Macro F1")
plt.tight_layout()
plt.savefig(
    os.path.join(save_root, "client_rf_macro_f1.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nSaved:")
print(summary_path)
print(os.path.join(save_root, "client_rf_macro_f1.png"))