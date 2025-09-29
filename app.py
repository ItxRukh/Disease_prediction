from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# --- Model & Features Load ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)   # training ke waqt ke symptom features list

# --- Data for description & precautions ---
desc = pd.read_csv("symptom_Description.csv")
prec = pd.read_csv("symptom_precaution.csv")

# --- Helper functions ---
def get_description(disease):
    row = desc[desc['Disease'] == disease]
    return row['Description'].values[0] if not row.empty else "No description available."

def get_precautions(disease):
    row = prec[prec['Disease'] == disease]
    if not row.empty:
        return [row[f'Precaution_{i}'].values[0] for i in range(1, 5)]
    return ["No precautions found"]

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")
@app.route("/result", methods=["POST"])
def result():
    prediction = None
    description = None
    precautions = []
    score = None   # <-- new variable

    if request.method == "POST":
        # User input
        symptoms = request.form.get("symptoms")
        symptoms = [s.strip().lower() for s in symptoms.split(",")]

        # --- Convert symptoms into feature vector ---
        X_input = [[0] * len(features)]
        for s in symptoms:
            if s in features:
                X_input[0][features.index(s)] = 1

        # --- Prediction ---
        predicted_disease = model.predict(X_input)[0]
        prediction = predicted_disease

        # --- Confidence score (probability) ---
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            score = round(max(proba) * 100, 2)   # percentage with 2 decimal places
        else:
            score = "Not available"

        description = get_description(predicted_disease)
        precautions = get_precautions(predicted_disease)

    return render_template(
        "result.html",
        prediction=prediction,
        description=description,
        precautions=precautions,
        score=score
    )


# --- New Route: Suggestions ---
@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").lower()
    matches = [s for s in features if query in s.lower()]
    return jsonify(matches[:5])   # sirf top 5 suggestions bhejo
@app.route("/contact")
def contact():
    return render_template("contact.html")
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
