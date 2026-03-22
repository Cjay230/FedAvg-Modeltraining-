import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------------- PATHS ----------------
data_file = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\kaust_multiclass_dataset_clean.csv"
save_root = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local"

rf_dir = os.path.join(save_root, "kaust_rf_multiclass")
mlp_dir = os.path.join(save_root, "kaust_mlp_multiclass")
fig_dir = os.path.join(save_root, "figures")
metrics_dir = os.path.join(save_root, "metrics")

for path in [rf_dir, mlp_dir, fig_dir, metrics_dir]:
    os.makedirs(path, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("Loading cleaned KAUST dataset...")
df = pd.read_csv(data_file, low_memory=False)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

# remove rows with missing labels
df = df.dropna(subset=["label"]).copy()

# force all feature columns to numeric
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill remaining NaNs with 0
df = df.fillna(0)

X = df.drop(columns=["label"]).copy()
y = df["label"].astype(str).copy()

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])
print("Classes:")
print(y.value_counts())

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

split_summary = pd.DataFrame({
    "split": ["train", "test"],
    "rows": [len(X_train), len(X_test)]
})
split_summary.to_csv(os.path.join(save_root, "train_test_split_summary.csv"), index=False)

# ---------------- HELPERS ----------------
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

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_predictions(y_true, y_pred, save_dir):
    pred_df = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred
    })
    pred_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

def save_confusion_and_report(model_name, y_true, y_pred, labels, save_dir, fig_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    save_json(report, os.path.join(save_dir, "classification_report.json"))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"{model_name.lower()}_confusion_matrix.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

def run_rf(model_name, model, save_dir):
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    save_json(metrics, os.path.join(metrics_dir, f"{model_name.lower()}_metrics.json"))

    save_predictions(y_test, y_pred, save_dir)
    save_confusion_and_report(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        labels=sorted(y.unique()),
        save_dir=save_dir,
        fig_dir=fig_dir
    )

    print(f"{model_name} metrics:")
    print(metrics)
    return metrics

def run_mlp_sampled(model_name, model, save_dir, sample_n=200000):
    # sample smaller subset for memory-safe MLP training
    actual_n = min(sample_n, len(X_train))
    sample_idx = X_train.sample(n=actual_n, random_state=42, replace=False).index
    X_train_mlp = X_train.loc[sample_idx].copy()
    y_train_mlp = y_train.loc[sample_idx].copy()

    # enforce numeric again just in case
    X_train_mlp = X_train_mlp.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test_mlp = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"\nTraining {model_name} on sampled subset: {len(X_train_mlp)} rows")

    model.fit(X_train_mlp, y_train_mlp)
    y_pred = model.predict(X_test_mlp)

    metrics = compute_metrics(y_test, y_pred)
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    save_json(metrics, os.path.join(metrics_dir, f"{model_name.lower()}_metrics.json"))

    save_predictions(y_test, y_pred, save_dir)
    save_confusion_and_report(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        labels=sorted(y.unique()),
        save_dir=save_dir,
        fig_dir=fig_dir
    )

    print(f"{model_name} metrics:")
    print(metrics)
    return metrics

# ---------------- MODELS ----------------
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

mlp_model = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=20,
        random_state=42,
        early_stopping=False
    ))
])

# ---------------- RUN EXPERIMENTS ----------------
rf_metrics = run_rf("RF", rf_model, rf_dir)
mlp_metrics = run_mlp_sampled("MLP", mlp_model, mlp_dir, sample_n=200000)

# ---------------- MODEL COMPARISON ----------------
comparison_df = pd.DataFrame([
    {"model": "RF", **rf_metrics},
    {"model": "MLP", **mlp_metrics},
])

comparison_df.to_csv(os.path.join(save_root, "model_comparison.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.bar(comparison_df["model"], comparison_df["macro_f1"])
plt.title("Model Comparison - Macro F1")
plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.tight_layout()
plt.savefig(
    os.path.join(fig_dir, "model_comparison_macro_f1.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nSaved outputs:")
print("- model_comparison.csv")
print("- train_test_split_summary.csv")
print("- per-model metrics / predictions / reports")
print("- confusion matrix figures")
print("- model_comparison_macro_f1.png")