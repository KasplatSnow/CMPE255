# Import necessary libraries
from flask import Flask, render_template_string, request
import pandas as pd
from io import StringIO # For handling file stream from upload
import os
import tempfile # For creating temporary files for the uploaded CSV

try:
    # Assumes interface.py is run as part of the 'src' package
    from .demo import classify_user_with_pipeline
except ImportError:
    try:
        from demo import classify_user_with_pipeline
    except ImportError as e:
        raise ImportError(
            "Could not import 'classify_user_with_pipeline' function from 'demo.py'. "
            "Ensure demo.py exists in the src directory and contains "
            "this function. Original error: {}".format(e)
        )

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Default model to use in the interface (e.g., 'mlp' since it was just trained)
DEFAULT_MODEL_NAME = "mlp" 
DEFAULT_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "results", "models", f"{DEFAULT_MODEL_NAME}_pipeline.pkl")
DEFAULT_ENCODER_PATH = os.path.join(PROJECT_ROOT, "results", "models", f"{DEFAULT_MODEL_NAME}_label_encoder.pkl")

# --- HTML Templates (using Tailwind CSS for styling) ---
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keystroke Classification</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .file-input-button::file-selector-button {
            margin-right: 0.5rem; padding: 0.5rem 1rem;
            border-radius: 0.375rem; border-width: 0px;
            font-size: 0.875rem; line-height: 1.25rem; font-weight: 600;
            color: #4f46e5; /* indigo-700 */
            background-color: #e0e7ff; /* indigo-50 */
        }
        .file-input-button::file-selector-button:hover {
            background-color: #c7d2fe; /* indigo-100 */
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
        <h1 class="text-3xl font-bold mb-8 text-center text-gray-800">Keystroke ID Validator</h1>
        <form method="post" action="/submit" enctype="multipart/form-data">
            <div class="mb-6">
                <label for="username" class="block text-md font-medium text-gray-700 mb-2">Claimed Username:</label>
                <input type="text" id="username" name="username" placeholder="e.g., s002" required
                       class="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-150">
            </div>
            <div class="mb-8">
                <label for="file" class="block text-md font-medium text-gray-700 mb-2">Upload Keystroke Data (CSV):</label>
                <input type="file" id="file" name="file" accept=".csv" required
                       class="w-full text-sm text-gray-500 file-input-button
                              border border-gray-300 rounded-lg cursor-pointer focus:outline-none">
            </div>
            <button type="submit"
                    class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out text-lg shadow-md hover:shadow-lg">
                Validate Keystrokes
            </button>
        </form>
        {% if error %}
            <div class="mt-6 p-4 bg-red-100 text-red-700 border border-red-300 rounded-lg">
                <p class="font-medium">Error:</p>
                <p>{{ error }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Result</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-lg shadow-xl w-full max-w-md text-center">
        <h1 class="text-3xl font-bold mb-6 text-gray-800">Classification Result</h1>
        <div class="mb-4">
            <p class="text-lg text-gray-600">Claimed Username:</p>
            <p class="text-2xl font-semibold text-gray-900">{{ username }}</p>
        </div>
        <div class="mb-6">
            <p class="text-lg text-gray-600">Predicted User by Model:</p>
            <p class="text-2xl font-semibold text-indigo-600">{{ predicted_user }}</p>
        </div>
        {# --- CORRECTED CONDITION: Use result_status to determine color --- #}
        <div class="p-6 rounded-lg shadow-inner {{ 'bg-green-100 text-green-800 border-green-300' if result_status == 'Valid User' else 'bg-red-100 text-red-800 border-red-300' }} border">
            <p class="text-2xl font-bold">{{ result_status }}</p>
            <p class="mt-1 text-sm">{{ result_message }}</p>
        </div>
        <a href="/"
           class="mt-8 inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out text-lg shadow-md hover:shadow-lg">
            Try Another
        </a>
    </div>
</body>
</html>
"""

# Main page route
@app.route("/")
def home():
    """Renders the main upload form."""
    return render_template_string(HOME_TEMPLATE)

# Submit route for handling file upload and classification
@app.route("/submit", methods=["POST"])
def submit():
    """Handles file upload, calls classification using the unified pipeline, and shows the result."""
    username_claimed = request.form.get("username")
    file = request.files.get("file")
    error_message = None

    if not username_claimed:
        error_message = "Username is required. Please enter the username associated with the keystroke data."
        return render_template_string(HOME_TEMPLATE, error=error_message)

    if not file:
        error_message = "No file uploaded. Please select a CSV file containing keystroke data."
        return render_template_string(HOME_TEMPLATE, error=error_message)

    if not file.filename.lower().endswith('.csv'):
        error_message = "Invalid file type. Please upload a CSV file."
        return render_template_string(HOME_TEMPLATE, error=error_message)

    temp_csv_path = None # Initialize path variable
    try:
        file_content = file.stream.read()
        if not file_content:
            error_message = "The uploaded CSV file is empty."
            return render_template_string(HOME_TEMPLATE, error=error_message)

        # Create a named temporary file to store the CSV data.
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv', dir='.') as tmp_file:
            tmp_file.write(file_content)
            temp_csv_path = tmp_file.name 
        
        print(f"INTERFACE: Temporary CSV for prediction saved at: {temp_csv_path}")
        print(f"INTERFACE: Using pipeline: {DEFAULT_PIPELINE_PATH}")
        print(f"INTERFACE: Using encoder: {DEFAULT_ENCODER_PATH}")

        # --- Call the classification function from demo.py ---
        predicted_user_label = classify_user_with_pipeline(
            input_csv_path=temp_csv_path,
            pipeline_path=DEFAULT_PIPELINE_PATH,
            encoder_path=DEFAULT_ENCODER_PATH
        )

        # Determine the result based on comparison
        result_status = "Valid User" if str(predicted_user_label) == str(username_claimed) else "Invalid User"
        result_message = (f"The keystroke pattern matches user '{username_claimed}'." 
                          if result_status == "Valid User" 
                          else f"The keystroke pattern does NOT match user '{username_claimed}'. Model identified it as '{predicted_user_label}'.")
        
        return render_template_string(RESULT_TEMPLATE,
                                      username=username_claimed,
                                      predicted_user=predicted_user_label,
                                      result_status=result_status,
                                      result_message=result_message)

    except FileNotFoundError as e:
        print(f"INTERFACE ERROR (FileNotFoundError): {e}")
        error_message = (f"A required model file was not found. Please ensure the model '{DEFAULT_MODEL_NAME}' "
                         "has been trained and its pipeline/encoder files exist. Details: {e}")
        return render_template_string(HOME_TEMPLATE, error=error_message)
    except ValueError as e: # Catch ValueErrors from demo.py (e.g., no features, empty df)
        print(f"INTERFACE ERROR (ValueError): {e}")
        error_message = f"Error during processing or prediction: {e}"
        return render_template_string(HOME_TEMPLATE, error=error_message)
    except RuntimeError as e: # Catch RuntimeErrors from demo.py (e.g., loading pkl fails)
        print(f"INTERFACE ERROR (RuntimeError): {e}")
        error_message = f"Runtime error during processing: {e}"
        return render_template_string(HOME_TEMPLATE, error=error_message)
    except Exception as e:
        print(f"INTERFACE ERROR (Unexpected): {e}")
        import traceback
        traceback.print_exc() 
        error_message = "An unexpected error occurred. Please check server logs or contact support."
        return render_template_string(HOME_TEMPLATE, error=error_message)
    finally:
        # Clean up the temporary file
        if temp_csv_path and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
                print(f"INTERFACE: Temporary CSV file deleted: {temp_csv_path}")
            except Exception as e_del:
                print(f"INTERFACE ERROR: Could not delete temporary file {temp_csv_path}: {e_del}")


if __name__ == "__main__":
    # Basic checks for model files at startup
    if not os.path.exists(DEFAULT_PIPELINE_PATH):
         print(f"WARNING (Interface Startup): Default pipeline file NOT FOUND at {DEFAULT_PIPELINE_PATH}")
         print(f"Please train the '{DEFAULT_MODEL_NAME}' model using 'python src/train.py --model {DEFAULT_MODEL_NAME}'")
    if not os.path.exists(DEFAULT_ENCODER_PATH):
         print(f"WARNING (Interface Startup): Default label encoder file NOT FOUND at {DEFAULT_ENCODER_PATH}")
    
    app.run(debug=True) # debug=True is useful for development
