# app.py

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    message = ""

    if request.method == "POST":
        message = request.form.get("message")
        if message:
            output = model.predict([message])
            if output[0] == 0:
                result = "✅ This Message is NOT a Spam Message."
            else:
                result = "🚫 This Message is a SPAM Message."

    return render_template("index.html", result=result, message=message)

if __name__ == "__main__":
    app.run(debug=True)
