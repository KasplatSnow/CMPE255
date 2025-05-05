from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    """Clips features to a column-wise upper percentile to reduce extreme outliers."""

    def __init__(self, clip_percentile=0.99):
        self.clip_percentile = clip_percentile

    def fit(self, X, y=None):
        self.upper_bounds_ = X.quantile(self.clip_percentile)
        return self

    def transform(self, X):
        return X.clip(upper=self.upper_bounds_, axis=1)

class _KeystrokeFeatureBuilder(BaseEstimator, TransformerMixin):
    """Extracts only H./DD./UD. features and returns feature matrix with labels and groups."""

    def fit(self, X: pd.DataFrame, y: None = None):
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        feature_cols = [col for col in X.columns if col.startswith(('H.', 'DD.', 'UD.'))]
        self.feature_cols = feature_cols
        feats = X[feature_cols]
        labels = X["subject"].values
        groups = X["subject"].values
        print("Extracted feature matrix:", feats.shape)
        return feats, labels, groups

def build_preprocess_pipeline() -> Pipeline:
    """Returns full preprocessing pipeline."""

    return Pipeline(
        steps=[
            ("feats", _KeystrokeFeatureBuilder()),
            ("clip", ClipOutliersTransformer()),
            ("scaler", StandardScaler()),
            ("var_filter", VarianceThreshold(threshold=0.0005)),
            # ("pca", PCA(n_components=0.95))
        ]
    )

def load_dataset(path: str | Path, top_n_users: int | None = None):
    """Reads CSV and returns (X, y, groups) after full preprocessing pipeline."""

    df = pd.read_csv(path)

    if top_n_users:
        top_subjects = df["subject"].value_counts().nlargest(top_n_users).index
        df = df[df["subject"].isin(top_subjects)]

    pipe = build_preprocess_pipeline()
    X, y, groups = pipe["feats"].transform(df)
    X = pipe["clip"].fit_transform(X)
    X = pipe["scaler"].fit_transform(X)
    X = pipe["var_filter"].fit_transform(X)
    # X = pipe["pca"].fit_transform(X)
    return X, y, groups
