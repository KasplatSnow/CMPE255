"""Training script for keystroke-based user identification using selected ML models.

Usage:
    python src/train.py --model xgb
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from models import get_model
from preprocessing import load_dataset
from utils import get_logger, save_metrics, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(description="Keystroke‑ID training script")
    parser.add_argument("--data", default="data/DSL-StrongPasswordData.csv")
    parser.add_argument("--model", default="xgb", choices=["knn", "logreg", "rf", "xgb", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_out", default="results/metrics.csv")
    args = parser.parse_args()

    logger = get_logger("train")
    set_seed(args.seed)

    logger.info("📥  Loading, filtering top-10 users & preprocessing dataset …")
    X, y_raw, groups = load_dataset(args.data, top_n_users=10)

    # Label encoding across the full dataset
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logger.info("✔️  Feature matrix shape: %s", X.shape)

    clf_classes = np.unique(y)
    ys_pred, ys_true, probas_all = [], [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        logger.info("🗂  Fold %d", fold + 1)
        clf = get_model(args.model)
        clf.fit(X[train_idx], y[train_idx])

        raw_probas = clf.predict_proba(X[test_idx])
        fixed_probas = np.zeros((len(test_idx), len(clf_classes)))
        for idx, cls in enumerate(clf.classes_):
            cls_index = np.where(clf_classes == cls)[0][0]
            fixed_probas[:, cls_index] = raw_probas[:, idx]

        preds = np.argmax(fixed_probas, axis=1)
        ys_pred.append(preds)
        ys_true.append(y[test_idx])
        probas_all.append(fixed_probas)

    y_true = np.concatenate(ys_true)
    y_pred = np.concatenate(ys_pred)
    y_proba = np.vstack(probas_all)
    y_true_bin = label_binarize(y_true, classes=np.arange(y_proba.shape[1]))

    metrics = {
        "model": args.model,
        "acc":  accuracy_score(y_true, y_pred),
        "f1":   f1_score(y_true, y_pred, average="micro"),
        "auc":  roc_auc_score(y_true_bin, y_proba, multi_class="ovo"),
    }

    logger.info("📊  %s", metrics)
    save_metrics(metrics, Path(args.metrics_out))

    # Confusion Matrix Plot
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, cmap="YlGnBu", annot=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"results/figures/confmat_{args.model}.png", dpi=300)
        logger.info("🖼  Confusion matrix saved to results/figures/confmat_%s.png", args.model)
    except Exception as e:
        logger.warning("Could not generate confusion matrix plot: %s", e)

if __name__ == "__main__":
    main()
