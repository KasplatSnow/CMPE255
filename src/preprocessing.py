from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Clips features to a column-wise upper percentile to reduce extreme outliers.
    This transformer expects pandas DataFrames as input and will output DataFrames.
    """

    def __init__(self, clip_percentile=0.99):
        self.clip_percentile = clip_percentile
        self.upper_bounds_ = None
        self.feature_names_in_ = None # To store feature names seen during fit

    def fit(self, X, y=None):
        """
        Fits the transformer by computing the upper bounds for clipping based on X.
        Args:
            X (pd.DataFrame): Input features. Must be a pandas DataFrame.
            y : Ignored.
        Returns:
            self: The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ClipOutliersTransformer expects a pandas DataFrame input for fit.")
        
        X_df = X # Input is already a DataFrame
        
        if not X_df.empty:
            self.upper_bounds_ = X_df.quantile(self.clip_percentile)
            self.feature_names_in_ = X_df.columns.tolist() # Store feature names
        else:
            # Handle empty DataFrame case: no bounds can be computed.
            self.upper_bounds_ = pd.Series(dtype=float) # Store an empty Series
            self.feature_names_in_ = []
            print("Warning: ClipOutliersTransformer.fit called on an empty DataFrame. No upper bounds computed.")
        return self

    def transform(self, X):
        """
        Transforms the data by clipping features to the pre-computed upper bounds.
        Args:
            X (pd.DataFrame): Input features. Must be a pandas DataFrame.
        Returns:
            pd.DataFrame: Transformed features with outliers clipped.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ClipOutliersTransformer expects a pandas DataFrame input for transform.")
        
        X_df = X.copy() # Work on a copy to avoid modifying the original DataFrame

        if self.upper_bounds_ is None:
            # This case implies fit was not called or failed.
            raise RuntimeError("ClipOutliersTransformer.transform called before fit or fit failed to set upper_bounds_.")

        if self.upper_bounds_.empty:
            print("Warning: ClipOutliersTransformer.transform called with no effective bounds (fit on empty data or no features). Returning original data.")
            return X_df
        
        # Align upper_bounds_ to the columns of the current X_df.
        # Columns in X_df not seen during fit will not be clipped (fillna(np.inf)).
        # Columns seen during fit but not in X_df will be ignored by reindex.
        aligned_upper_bounds = self.upper_bounds_.reindex(X_df.columns).fillna(np.inf)
        
        return X_df.clip(upper=aligned_upper_bounds, axis=1)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for outputted data."""
        if input_features is None:
            if self.feature_names_in_ is None:
                raise ValueError("Cannot determine output feature names before fit or if fit on nameless data.")
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(input_features, dtype=object)


def load_raw_keystroke_data(path: str | Path, top_n_users: int | None = None, required_subject_column: str = "subject"):
    """
    Reads CSV, optionally filters for top N users, and extracts raw feature columns
    (starting with 'H.', 'DD.', 'UD.') and subject labels.
    No other preprocessing (scaling, clipping, etc.) is performed by this function.

    Args:
        path (str | Path): Path to the CSV file.
        top_n_users (int | None, optional): If specified, filters for the top N users
                                            based on session count. Defaults to None (all users).
        required_subject_column (str): The name of the column containing user/subject identifiers.

    Returns:
        pd.DataFrame: Raw features (X) as a DataFrame.
        pd.Series: Labels (y) corresponding to the subject column.
        pd.Series: Groups (typically the same as y for user classification tasks),
                   useful for StratifiedKFold if needed, though the main pipeline handles this.
                   
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV cannot be read, the subject column is missing, 
                    no feature columns are found, or data becomes empty.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at {data_path}: {e}")

    if required_subject_column not in df.columns:
        raise ValueError(f"Required subject column '{required_subject_column}' not found in CSV.")

    if top_n_users is not None and top_n_users > 0:
        top_subjects = df[required_subject_column].value_counts().nlargest(top_n_users).index
        df = df[df[required_subject_column].isin(top_subjects)].copy() # Use .copy()
        if df.empty:
            raise ValueError(f"DataFrame is empty after filtering for top {top_n_users} users. "
                             "Check dataset or top_n_users value.")
        print(f"Filtered for top {top_n_users} users. Resulting data shape: {df.shape}")

    # Extract feature columns (those starting with H., DD., or UD.)
    feature_cols = [col for col in df.columns if col.startswith(('H.', 'DD.', 'UD.'))]
    
    if not feature_cols:
        # If no such columns, it's an issue with the input data format for this specific task.
        raise ValueError("No feature columns (starting with 'H.', 'DD.', or 'UD.') found in the dataset. "
                         "Please check the CSV file format. Expected columns for keystroke features are missing.")
    
    X = df[feature_cols]
    y = df[required_subject_column]
    groups = df[required_subject_column] # For StratifiedKFold, if used outside a pipeline that handles it.

    print(f"Loaded raw data: X shape {X.shape}, y unique values {y.nunique()}")
    
    if X.empty:
        raise ValueError("Feature DataFrame X is empty after selecting H./DD./UD. columns. Check CSV content and feature column names.")
    if y.empty:
        raise ValueError(f"Labels Series y is empty. Check '{required_subject_column}' column in CSV.")

    return X, y, groups
