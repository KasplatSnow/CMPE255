import pandas as pd
import os

def extract_single_user_entry_features_to_csv(main_csv_path: str, subject_id: str, output_dir: str = ".") -> str | None:
    """
    Extracts the feature columns for the first data entry of a specific subject
    from a main CSV dataset and saves it to a new CSV file.
    The output CSV will only contain columns starting with 'H.', 'DD.', or 'UD.' and their header.

    Args:
        main_csv_path (str): Path to the main dataset CSV file (e.g., "DSL-StrongPasswordData.csv").
        subject_id (str): The ID of the subject whose first entry's features are to be extracted (e.g., "s015").
        output_dir (str): Directory where the new CSV file will be saved. Defaults to current directory.

    Returns:
        str | None: The path to the created CSV file if successful, otherwise None.
    """
    try:
        # Load the main dataset
        print(f"Loading main dataset from: {main_csv_path}")
        df_main = pd.read_csv(main_csv_path)
        print(f"Main dataset loaded successfully. Shape: {df_main.shape}")

        # Check if 'subject' column exists
        if 'subject' not in df_main.columns:
            print(f"Error: 'subject' column not found in {main_csv_path}. Please ensure it's the correct dataset.")
            return None

        # Filter data for the specified subject_id
        print(f"Filtering data for subject: {subject_id}")
        df_subject_all_entries = df_main[df_main['subject'] == subject_id]

        if df_subject_all_entries.empty:
            print(f"No data found for subject '{subject_id}' in the dataset.")
            return None

        # Take only the first entry for the subject
        df_single_entry_full = df_subject_all_entries.head(1)
        
        if df_single_entry_full.empty:
            print(f"Could not extract the first entry for subject '{subject_id}'.") # Should not happen
            return None

        feature_cols = [col for col in df_single_entry_full.columns if col.startswith(('H.', 'DD.', 'UD.'))]
        
        if not feature_cols:
            print(f"Error: No feature columns (starting with 'H.', 'DD.', 'UD.') found for subject '{subject_id}'s entry.")
            print("This might indicate an issue with the source data's column naming for this specific entry.")
            return None
            
        df_features_only = df_single_entry_full[feature_cols]

        print(f"Extracted features for the first entry of subject '{subject_id}'. Shape: {df_features_only.shape}")
        print(f"Feature columns: {df_features_only.columns.tolist()}")

        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{subject_id}_keystroke_features_single_entry.csv"
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the feature-only data to the new CSV file, including the header
        print(f"Saving features for subject '{subject_id}' (single entry) to: {output_filepath}")
        df_features_only.to_csv(output_filepath, index=False, header=True)
        print(f"Successfully created '{output_filepath}'.")
        
        return output_filepath

    except FileNotFoundError:
        print(f"Error: The main dataset file was not found at '{main_csv_path}'.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The main dataset file at '{main_csv_path}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main_dataset_path = "data/DSL-StrongPasswordData.csv" # Adjust this path as needed

    # User ID whose first entry's features we want to extract
    user_to_extract = "s042"

    # Directory to save the new CSV file
    output_directory = "."

    print(f"--- Starting Single User Entry Feature Extraction ---")
    print(f"Dataset: {main_dataset_path}")
    print(f"User to extract (first entry, features only): {user_to_extract}")
    print(f"Output directory: {os.path.abspath(output_directory)}")
    
    if not os.path.exists(main_dataset_path):
        print(f"\nERROR: Main dataset '{main_dataset_path}' not found.")
        print("Please ensure the path is correct or place the dataset in the expected location.")
    else:
        created_file = extract_single_user_entry_features_to_csv(main_dataset_path, user_to_extract, output_directory)

        if created_file:
            print(f"\nExtraction complete. Single entry (features only) test file for user '{user_to_extract}' is ready at: {created_file}")
        else:
            print(f"\nExtraction failed for user '{user_to_extract}'. Please check error messages above.")
    
    print(f"\n--- Script Finished ---")

