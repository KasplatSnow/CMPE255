"""
Model factory that returns a scikit-learn `Pipeline` with optional scaler + PCA + classifier.

Each model is wrapped to allow consistent usage with fit/predict_proba across scripts.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Common preprocessing steps used across all models.
# PCA is kept here for legacy compatibility, but may be removed if preprocessing handles it.
_common = [
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=0.95)),
]


def get_model(name: str) -> Pipeline:
    name = name.lower()
    if name == "knn":
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=750, n_jobs=-1, multi_class="multinomial")
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced")
    elif name == "xgb":
        clf = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from knn|logreg|rf|xgb|mlp.")

    return Pipeline(_common + [("clf", clf)])