import os
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from collections import defaultdict


EVALUE_DIR = "result/folddisco_results_evalues"
OUTPUT_PATH = f"{EVALUE_DIR}/total_evalues_test.txt"
PNG_DIR = "result/analyses_plot"
LOG_FILE = f"{EVALUE_DIR}/analysis_log.txt"
CUTOFF = 0.5  # Threshold for binary classification
AUC_ROC_FILE = f"{PNG_DIR}/roc_curve.png"
AUC_PR_FILE = f"{PNG_DIR}/precision_recall_curve.png"
CONFUSION_MATRIX_FILE = f"{PNG_DIR}/confusion_matrix.png"
RESULTS_FILE = f"{EVALUE_DIR}/evaluation_results.txt"

os.makedirs(PNG_DIR, exist_ok=True)

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def safe_float(x: str) -> float:
    try:
        return float(x)
    except:
        return float("nan")

def load_rows(path: str):
    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split()
        # Expect tab/space separated; your example looks tabbed but split() covers both.

        # Map column name -> index (robust to order)
        col = {name: i for i, name in enumerate(header)}

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            # If the file is space-separated instead of tab-separated:
            if len(parts) == 1:
                parts = line.strip().split()

            q = parts[col["query_num"]]
            tgt = parts[col["target_id"]]
            length = int(parts[col["L"]])
            idf = safe_float(parts[col["idf"]])
            rmsd = safe_float(parts[col["rmsd"]])
            tm_score = safe_float(parts[col["tm_score"]])
            label = int(parts[col["label_raw"]])
            score = safe_float(parts[col["score_logit"]])
            prob = safe_float(parts[col["prob_homologous"]])

            # Drop pathological rows (optional)
            if any(math.isnan(v) for v in [idf, rmsd, tm_score]) or length <= 0 or length > 32:
                continue

            rows.append({
                "query_num": q,
                "target_id": tgt,
                "L": length,
                "label_raw": label,
                "score_logit": score,
                "prob_homologous": prob
            })
    return rows

# Load data
rows = load_rows(OUTPUT_PATH)
log_and_print(f"Loaded {len(rows)} rows from {OUTPUT_PATH}")

# ---------- 1) Performance Evaluation ----------

# Extracting true labels and predicted scores
y_true = [1 if r["label_raw"] > 0 else 0 for r in rows]
y_scores = [r["prob_homologous"] for r in rows]  # Use logits (or probabilities if you prefer)

for i in y_true:
    if i != 0:
        i = 1

for score in y_scores:
    if math.isnan(score) or score == 0.0:
        score = 1e-10  # Avoid log(0) or NaN issues

# Calculate Accuracy
accuracy = accuracy_score(y_true, [1 if score > CUTOFF else 0 for score in y_scores])
log_and_print(f"Accuracy: {accuracy:.4f}")

# ---------- 2) ROC and AUC Curve ----------

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)
log_and_print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(AUC_ROC_FILE)
plt.close()

# ---------- 3) Precision-Recall Curve and AUC ----------

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)
log_and_print(f"Average Precision: {average_precision:.4f}")

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig(AUC_PR_FILE)
plt.close()

# ---------- 4) Confusion Matrix ----------

# Binarize predictions (0 or 1)
y_pred_bin = [1 if score > CUTOFF else 0 for score in y_scores]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_bin)
log_and_print(f"Confusion Matrix:\n{cm}")

# Plot confusion matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Non-homologous', 'Homologous']
tick_marks = [0, 1]
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE)
plt.close()

# ---------- 5) Precision, Recall, and F1 Score ----------

# Precision and Recall
precision_score = precision_score(y_true, y_pred_bin)
recall_score = recall_score(y_true, y_pred_bin)
f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)

log_and_print(f"Precision: {precision_score:.4f}")
log_and_print(f"Recall: {recall_score:.4f}")
log_and_print(f"F1 Score: {f1:.4f}")

# ---------- 6) Save Results to File ----------

# Save final results (e.g., accuracy, precision, etc.)
with open(RESULTS_FILE, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Average Precision: {average_precision:.4f}\n")
    f.write(f"Precision: {precision_score:.4f}\n")
    f.write(f"Recall: {recall_score:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

log_and_print("Evaluation complete. Results saved to evaluation_results.txt.")
