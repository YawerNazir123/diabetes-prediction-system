from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")


# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT ROUTE ----------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- MODEL DETAILS ROUTE ----------------
@app.route("/model")
def model_info():
    return render_template("model.html")


# ---------------- PREDICTION ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        prediction = model.predict([features])[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
