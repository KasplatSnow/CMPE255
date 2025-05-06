from flask import Flask, render_template_string, request
import pandas as pd
from io import StringIO
from demo import classify_user_demo

app = Flask(__name__)

#main page
@app.route("/")
def home():
    return render_template_string("""
        <html>
            <head><title>CSV Upload</title></head>
            <body>
                <h1>Classify by Keystroke</h1>
                <form method="post" action="/submit" enctype="multipart/form-data">
                    <input type="text" name="username" placeholder="Enter username" required /><br><br>
                    <input type="file" name="file" accept=".csv" required /><br><br>
                    <input type="submit" value="Validate" />
                </form>
            </body>
        </html>
    """)

#'submit' button function
@app.route("/submit", methods=["POST"])
def submit():
    username = request.form.get("username")
    file = request.files.get("file")

    if not file or not file.filename.endswith('.csv'):
        return "<h2>Error: Please upload a valid CSV file.</h2>"

    csv_data = StringIO(file.stream.read().decode("utf-8"))
    df = pd.read_csv(csv_data)

    #call a model for classifcation
    #Needs a proper function to perform classification
    #should return a prediucted user by the classifier
    predicted_user = classify_user_demo(username, df)

    result = "Valid" if predicted_user == username else "Invalid"

    #
    return f"""
        <html>
            <head><title>Result</title></head>
            <body>
                <h2>Result: {result}</h2>
                <a href="/">Try again</a>
            </body>
        </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
