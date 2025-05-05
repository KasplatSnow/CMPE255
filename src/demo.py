import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from preprocessing import ClipOutliersTransformer
from pathlib import Path

def classify_user_demo(username: str, input_csv_path: str,
                       model_path: str = "results/models/mlp_model.pkl",
                       encoder_path: str = "results/models/mlp_label_encoder.pkl") -> str:
    """
    Predict the user from a keystroke sample and compare to the input username.

    Args:
        username (str): The claimed username.
        input_csv_path (str): Path to CSV file containing the keystroke sample(s).
        model_path (str): Path to the trained model file.
        encoder_path (str): Path to the saved LabelEncoder.
    
    Returns:
        str: Predicted username.
    """

    # Load model and label encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Load and preprocess the input sample
    df = pd.read_csv(input_csv_path)
    feature_cols = [col for col in df.columns if col.startswith(('H.', 'DD.', 'UD.'))]
    X = df[feature_cols]

    # Apply same preprocessing steps as training
    clipper = ClipOutliersTransformer()
    X = clipper.fit_transform(X)  # You can replace with a saved one if needed
    X = model.named_steps['scaler'].transform(X)
    if 'var_filter' in model.named_steps:
        X = model.named_steps['var_filter'].transform(X)

    # Predict
    y_pred = model.predict(X)[0]
    predicted_user = label_encoder.inverse_transform([y_pred])[0]

    print(f"Claimed username: {username}")
    print(f"Predicted username: {predicted_user}")

    return predicted_user
