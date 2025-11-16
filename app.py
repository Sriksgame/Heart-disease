# app.py  — serves index.html and the /predict API
from flask import Flask, request, jsonify, send_from_directory
import joblib, numpy as np
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='.')
CORS(app)  # allow fetch from index.html

# Load model, columns, encoders, imputer
model, columns, encoders, imputer = joblib.load("heart_model.pkl")

@app.route("/")
def index():
    # serve the index.html from the same folder
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    processed = []

    for col in columns:
        value = data.get(col, None)

        # If value is missing, use NaN placeholder (imputer will handle)
        if value is None or value == "":
            processed.append(np.nan)
            continue

        # Encode categorical using saved encoders
        if col in encoders:
            encoder = encoders[col]
            # encoder was trained on string values: transform expects the exact string used in training
            value = encoder.transform([str(value)])[0]
        else:
            try:
                value = float(value)
            except:
                # fallback to NaN if cannot convert
                value = np.nan

        processed.append(value)

    processed = np.array(processed).reshape(1, -1)
    processed = imputer.transform(processed)

    prob = model.predict_proba(processed)[0][1]
    label = "High Risk" if prob > 0.5 else "Low Risk"

    return jsonify({
        "probability": round(float(prob), 3),
        "result": label
    })

if __name__ == "__main__":
    # ensure you're running on port 5000, accessible at http://127.0.0.1:5000/
    app.run(host="127.0.0.1", port=5000, debug=True)
