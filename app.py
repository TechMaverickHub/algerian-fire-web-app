import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)  # Use only one Flask instance

# Load model & scaler (handle potential errors)
try:
    ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
    standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    ridge_model = None
    standard_scaler = None

@app.route("/")
def index():
    return render_template("index.html")  # Ensure "templates/index.html" exists

if __name__ == "__main__":
    app.run(debug=True)  # Run Flask in debug mode