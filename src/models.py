"""
Model factory that returns a scikit-learn `Pipeline` incorporating all
preprocessing steps (clipping, variance threshold, scaling, PCA) and a classifier.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .preprocessing import ClipOutliersTransformer


def get_model_pipeline(name: str, pca_n_components: float | int | None = 0.95, random_seed: int = 42) -> Pipeline:
    """
    Builds and returns a scikit-learn pipeline including preprocessing and a classifier.
    The input to this pipeline should be the raw feature DataFrame (H., DD., UD. columns).

    Args:
        name (str): The name of the classifier to use.
                    Choices: "knn", "logreg", "rf", "xgb", "mlp".
        pca_n_components (float | int | None, optional): Number of components for PCA.
            If float (e.g., 0.95), it's the variance to keep.
            If int, it's the number of components.
            If None, PCA step is skipped.
            Defaults to 0.95.
        random_seed (int): Seed for reproducibility for classifiers that support it.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    
    Raises:
        ValueError: If an unknown model name is provided.
    """
    name = name.lower()
    
    # Define the classifier based on the input name
    if name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    elif name == "logreg":
        clf = LogisticRegression(
            max_iter=1000, 
            solver='liblinear', 
            multi_class="auto", 
            random_state=random_seed, 
            n_jobs=-1
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, 
            class_weight="balanced", 
            random_state=random_seed, 
            n_jobs=-1
        )
    elif name == "xgb":
        clf = XGBClassifier(
            n_estimators=300, 
            learning_rate=0.1, 
            max_depth=5, 
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob", 
            eval_metric="mlogloss",    
            use_label_encoder=False, 
            random_state=random_seed,
            n_jobs=-1
        )
    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            activation="relu",
            solver='adam',
            batch_size='auto', 
            learning_rate_init=0.001, 
            max_iter=300, 
            early_stopping=True,
            n_iter_no_change=15, 
            random_state=random_seed
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from knn|logreg|rf|xgb|mlp.")

    # The input X to this pipeline should be the DataFrame with H., DD., UD. columns.
    pipeline_steps = [
        # The ClipOutliersTransformer instance will be pickled with its module path.
        # When run as `python -m src.train`, this path will be `src.preprocessing.ClipOutliersTransformer`
        ('clip', ClipOutliersTransformer(clip_percentile=0.99)),
        ('var_filter', VarianceThreshold(threshold=0.0005)), 
        ('scaler', StandardScaler())
    ]

    if pca_n_components is not None:
        pipeline_steps.append(('pca', PCA(n_components=pca_n_components, random_state=random_seed)))
    
    pipeline_steps.append(('clf', clf))
    
    full_pipeline = Pipeline(steps=pipeline_steps)
    
    return full_pipeline
