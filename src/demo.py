"""
Demo script for classifying a user based on keystroke data using a pre-trained
full pipeline (preprocessing + classifier).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib # For loading the pipeline and encoder

def classify_user_with_pipeline(
    input_csv_path: str | Path,
    pipeline_path: str | Path, # Path to the saved *fitted* pipeline
    encoder_path: str | Path   # Path to the saved label encoder
) -> str:
    """
    Predicts the user from a keystroke sample CSV using a saved, fitted pipeline.

    Args:
        input_csv_path (str | Path): Path to the CSV file containing the keystroke sample(s).
                                     This CSV must contain columns starting with 'H.', 'DD.', or 'UD.'
        pipeline_path (str | Path): Path to the saved, fitted scikit-learn pipeline file (.pkl).
        encoder_path (str | Path): Path to the saved LabelEncoder file (.pkl).
    
    Returns:
        str: The predicted username/subject.
    
    Raises:
        FileNotFoundError: If the pipeline, encoder, or input CSV file is not found.
        ValueError: If the input CSV does not yield any required features, or if other
                    prediction errors occur.
    """
    input_csv_path = Path(input_csv_path)
    pipeline_path = Path(pipeline_path)
    encoder_path = Path(encoder_path)

    # --- Validate file existence ---
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found at: {input_csv_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Fitted pipeline file not found at: {pipeline_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found at: {encoder_path}")

    print(f"DEMO: Loading pipeline from: {pipeline_path}")
    print(f"DEMO: Loading label encoder from: {encoder_path}")
    
    # --- Load the fitted pipeline and label encoder ---
    try:
        pipeline = joblib.load(pipeline_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline or encoder: {e}")

    # --- Load and prepare the input sample data ---
    try:
        df_input = pd.read_csv(input_csv_path)
    except Exception as e:
        raise ValueError(f"Could not read or parse the input CSV file at {input_csv_path}. Error: {e}")

    # Extract only the raw feature columns (H., DD., UD.)
    # The pipeline expects a DataFrame with these columns as input.
    feature_cols = [col for col in df_input.columns if col.startswith(('H.', 'DD.', 'UD.'))]
    
    if not feature_cols:
        raise ValueError(
            "No feature columns (starting with 'H.', 'DD.', or 'UD.') found in the input CSV. "
            "Please ensure the CSV format is correct."
        )
    
    X_input_raw = df_input[feature_cols]

    if X_input_raw.empty:
        raise ValueError(
            "The feature DataFrame (X_input_raw) is empty after selecting feature columns from the input CSV. "
            "This could be due to an empty CSV or a CSV with relevant column headers but no data rows."
        )
    
    print(f"DEMO: Raw input features X_input_raw shape for prediction: {X_input_raw.shape}")
    print(f"DEMO: Columns in X_input_raw: {X_input_raw.columns.tolist()}")

    # --- Predict using the loaded pipeline ---
    # The pipeline will internally handle all preprocessing steps (clipping, variance filtering, scaling, PCA)
    # using the parameters learned during its training.
    try:
        # `pipeline.predict()` returns the numerically encoded prediction(s).
        # Assuming the input CSV represents one or more samples for prediction.
        # If it's always one sample (one row or one set of session data),
        # y_pred_encoded will be an array with one element.
        y_pred_encoded = pipeline.predict(X_input_raw)
        
        if y_pred_encoded.size == 0:
            raise ValueError("Prediction resulted in an empty array. Check pipeline and input data.")

        # For a single sample prediction, take the first element.
        # If multiple samples are in the CSV, this would need to be handled accordingly
        # (e.g., predict for each row and decide how to aggregate or present).
        predicted_user_encoded = y_pred_encoded[0]
        
        # Convert the numeric prediction back to the original subject name/label
        predicted_user_label = label_encoder.inverse_transform([predicted_user_encoded])[0]
        
    except Exception as e:
        # Catch potential errors during pipeline.predict()
        # This could include issues if the input data structure, after internal transformations,
        # doesn't match what a later step in the pipeline expects (e.g., wrong number of features for clf).
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise ValueError(f"Error during pipeline prediction: {e}. Check data compatibility with the trained pipeline.")

    print(f"DEMO: Predicted user (encoded): {predicted_user_encoded}, Predicted user (label): {predicted_user_label}")
    return str(predicted_user_label) # Ensure string output


if __name__ == '__main__':
    
    print("Running demo.py directly for testing...")
    
    project_root = Path(__file__).resolve().parent.parent
    
    model_name = "mlp" 
    default_pipeline_path = project_root / "results" / "models" / f"{model_name}_pipeline.pkl"
    default_encoder_path = project_root / "results" / "models" / f"{model_name}_label_encoder.pkl"
    
    # Check if the default model files exist before trying to use them
    if not default_pipeline_path.exists() or not default_encoder_path.exists():
        print(f"Default model/encoder for '{model_name}' not found. Please train the model first using:")
        print(f"python src/train.py --model {model_name}")
    else:
        # --- Test with a sample CSV ---
        test_csv_path = project_root / "s015_keystroke.csv" # Make sure this file exists at this location

        if test_csv_path.exists():
            try:
                print(f"\nTesting with CSV: {test_csv_path}")
                # The 'username' argument is not used by classify_user_with_pipeline,
                # it's more for the web interface logic to compare against.
                # Here, we just want to see the prediction.
                predicted_subject = classify_user_with_pipeline(
                    input_csv_path=test_csv_path,
                    pipeline_path=default_pipeline_path,
                    encoder_path=default_encoder_path
                )
                print(f"Predicted Subject for {test_csv_path.name}: {predicted_subject}")
            except Exception as e:
                print(f"Error during direct test with {test_csv_path.name}: {e}")
        else:
            print(f"Test CSV file not found at {test_csv_path}. Skipping direct test with this file.")
            print("Please place a sample CSV (e.g., s015_keystroke.csv) in the project root or update the path.")

        # --- Test with another sample CSV ---
        test_csv_path_2 = project_root / "s042_keystroke.csv"
        if test_csv_path_2.exists():
            try:
                print(f"\nTesting with CSV: {test_csv_path_2}")
                predicted_subject_2 = classify_user_with_pipeline(
                    input_csv_path=test_csv_path_2,
                    pipeline_path=default_pipeline_path,
                    encoder_path=default_encoder_path
                )
                print(f"Predicted Subject for {test_csv_path_2.name}: {predicted_subject_2}")
            except Exception as e:
                print(f"Error during direct test with {test_csv_path_2.name}: {e}")
        else:
            print(f"Test CSV file not found at {test_csv_path_2}. Skipping direct test with this file.")

    print("\nDemo script direct execution finished.")
