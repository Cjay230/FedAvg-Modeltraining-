import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Since some transportation modes are much more frequent than others, 
# I trained a class-weighted Random Forest where minority classes are given more importance during training.
#  The goal was to test whether this improves fairness across classes compared to the standard Random Forest baseline.
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

data_file = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\kaust_multiclass_dataset_clean.csv"
save_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\kaust_rf_weighted"
fig_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\figures"
metrics_dir = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\metrics"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

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

print("Loading cleaned KAUST dataset...")
df = pd.read_csv(data_file, low_memory=False)

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
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

print("Training weighted RF...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = compute_metrics(y_test, y_pred)
print("Weighted RF metrics:")
print(metrics)

save_json(metrics, os.path.join(save_dir, "metrics.json"))
save_json(metrics, os.path.join(metrics_dir, "rf_weighted_metrics.json"))

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})
pred_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

report = classification_report(
    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
)
save_json(report, os.path.join(save_dir, "classification_report.json"))

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Weighted RF Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha="right")
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(
    os.path.join(fig_dir, "rf_weighted_confusion_matrix.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()

comparison_path = r"C:\Users\user\Desktop\VIP\federated\results\part2_kaust_local\model_comparison.csv"
if os.path.exists(comparison_path):
    comparison_df = pd.read_csv(comparison_path)
    comparison_df = comparison_df[comparison_df["model"] != "RF_weighted"]
    comparison_df = pd.concat([
        comparison_df,
        pd.DataFrame([{"model": "RF_weighted", **metrics}])
    ], ignore_index=True)
    comparison_df.to_csv(comparison_path, index=False)
    print("Updated model_comparison.csv")