"""
Training script for keystroke-based user identification.
This script loads raw features, builds a full preprocessing and classification pipeline,
trains it using cross-validation, evaluates it, and then saves the
final fitted pipeline and the label encoder.

Usage:
    python src/train.py --model mlp
    (Run from the project root directory, e.g., CMPE255/)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib 

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from .models import get_model_pipeline
from .preprocessing import load_raw_keystroke_data
from .utils import get_logger, save_metrics, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Keystroke User ID Training Script (Unified Pipeline)")
    parser.add_argument("--data", default="data/DSL-StrongPasswordData.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--model", default="mlp", choices=["knn", "logreg", "rf", "xgb", "mlp"], help="Classifier model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--metrics_out", default="results/metrics.csv", help="Path to save evaluation metrics CSV.")
    parser.add_argument("--top_n_users", type=int, default=10, help="Filter dataset for top N users. Set to 0 or None for all users.")
    parser.add_argument("--pca_components", type=float, default=0.95, help="PCA components to keep (e.g., 0.95 for 95% variance, or an int for number of components). Set to 0 or None to disable PCA.")
    args = parser.parse_args()

    logger = get_logger("train_pipeline_abs_imports") 
    set_seed(args.seed)

    top_n_users_param = args.top_n_users if args.top_n_users > 0 else None
    pca_components_param = args.pca_components if args.pca_components > 0 else None # Corrected from > 0.0

    logger.info(f"📥 Loading raw dataset from {args.data} for top {top_n_users_param or 'all'} users...")
    X_raw, y_raw_labels, groups = load_raw_keystroke_data(args.data, top_n_users=top_n_users_param)
    
    logger.info(f"✔️ Raw feature matrix X_raw shape: {X_raw.shape}")
    logger.info(f"✔️ Raw labels y_raw_labels unique count: {y_raw_labels.nunique()}")

    # Label encode the subject names (y_raw_labels) to integers for the model
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw_labels)
    logger.info(f"✔️ Encoded labels y_encoded shape: {y_encoded.shape}, unique encoded values: {np.unique(y_encoded).size}")

    # Store actual class names corresponding to encoded labels for confusion matrix plotting
    class_names = label_encoder.classes_

    # Initialize lists to store results from each fold
    fold_metrics = []
    all_y_true_fold = []
    all_y_pred_fold = []
    all_y_proba_fold = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    logger.info(f"🚀 Starting {cv.get_n_splits()}-fold cross-validation for model: {args.model} with PCA components: {pca_components_param}")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw, y_encoded, groups=groups)): 
        logger.info(f"Fold {fold + 1}/{cv.get_n_splits()}")

        # Get a new model pipeline instance for each fold
        pipeline = get_model_pipeline(args.model, pca_n_components=pca_components_param, random_seed=args.seed)

        X_train_fold, X_test_fold = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]

        logger.info(f"Fitting pipeline on X_train_fold (shape: {X_train_fold.shape}) and y_train_fold (shape: {y_train_fold.shape})")
        pipeline.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = pipeline.predict(X_test_fold)
        y_proba_fold = pipeline.predict_proba(X_test_fold)

        all_y_true_fold.extend(y_test_fold)
        all_y_pred_fold.extend(y_pred_fold)
        all_y_proba_fold.append(y_proba_fold)

        acc_fold = accuracy_score(y_test_fold, y_pred_fold)
        f1_fold = f1_score(y_test_fold, y_pred_fold, average="micro")
        y_test_fold_bin = label_binarize(y_test_fold, classes=np.arange(len(class_names)))
        auc_fold = roc_auc_score(y_test_fold_bin, y_proba_fold, multi_class="ovo", average="macro")
        
        fold_metrics.append({'fold': fold + 1, 'acc': acc_fold, 'f1': f1_fold, 'auc': auc_fold})
        logger.info(f"Fold {fold + 1} Metrics: Acc={acc_fold:.4f}, F1={f1_fold:.4f}, AUC={auc_fold:.4f}")

    y_true_overall = np.array(all_y_true_fold)
    y_pred_overall = np.array(all_y_pred_fold)
    y_proba_overall = np.vstack(all_y_proba_fold)

    overall_acc = accuracy_score(y_true_overall, y_pred_overall)
    overall_f1 = f1_score(y_true_overall, y_pred_overall, average="micro")
    y_true_overall_bin = label_binarize(y_true_overall, classes=np.arange(len(class_names)))
    overall_auc = roc_auc_score(y_true_overall_bin, y_proba_overall, multi_class="ovo", average="macro")

    final_metrics = {
        "model": args.model, "pca_setting": str(pca_components_param), 
        "top_n_users": str(top_n_users_param), "acc_cv_mean": np.mean([m['acc'] for m in fold_metrics]),
        "f1_cv_mean":  np.mean([m['f1'] for m in fold_metrics]), "auc_cv_mean": np.mean([m['auc'] for m in fold_metrics]),
        "acc_overall": overall_acc, "f1_overall":  overall_f1, "auc_overall": overall_auc,
    }
    logger.info(f"📊 Overall CV Aggregated Metrics: {final_metrics}")
    save_metrics(final_metrics, Path(args.metrics_out))

    logger.info(f"⚙️ Training final model ({args.model}) on the entire dataset...")
    final_pipeline = get_model_pipeline(args.model, pca_n_components=pca_components_param, random_seed=args.seed)
    final_pipeline.fit(X_raw, y_encoded)
    logger.info("✔️ Final model training complete.")

    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_save_path = models_dir / f"{args.model}_pipeline.pkl"
    encoder_save_path = models_dir / f"{args.model}_label_encoder.pkl"

    joblib.dump(final_pipeline, pipeline_save_path)
    joblib.dump(label_encoder, encoder_save_path)
    logger.info(f"💾 Final pipeline saved to: {pipeline_save_path}")
    logger.info(f"💾 Label encoder saved to: {encoder_save_path}")

    try:
        cm = confusion_matrix(y_true_overall, y_pred_overall, labels=np.arange(len(class_names)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for {args.model} (Overall CV)')
        plt.xlabel("Predicted Label"); plt.ylabel("True Label")
        figures_dir = Path("results/figures"); figures_dir.mkdir(parents=True, exist_ok=True)
        cm_save_path = figures_dir / f"confmat_{args.model}_pipeline.png" # Changed filename slightly
        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight'); plt.close()
        logger.info(f"🖼 Confusion matrix saved to {cm_save_path}")
    except Exception as e:
        logger.warning(f"Could not generate or save confusion matrix plot: {e}")

if __name__ == "__main__":
    main()
