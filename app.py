from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.exceptions import InconsistentVersionWarning
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
import logging

# Suppress scikit-learn warning about feature names
warnings.filterwarnings('ignore', message="X does not have valid feature names")
# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model, scaler, and label encoder
model = joblib.load('tax_revenue_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Initialize logging
logging.basicConfig(level=logging.INFO)


@app.route('/')
def home():
    return "RSL Tax Revenue Prediction API is running."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request JSON
        data = request.json

        # Extract input features from the data
        revenue_category = data.get('revenue_category')
        year = data.get('year')

        # Check for None or empty values
        if revenue_category is None or revenue_category == '':
            raise ValueError("Revenue category cannot be None or empty")
        if year is None:
            raise ValueError("Year cannot be None")

        # Encode the revenue category using LabelEncoder
        encoded_category = label_encoder.transform([revenue_category])[0]

        # Combine the encoded category with the year
        input_features = np.array([encoded_category, year])

        # Transform input features using the scaler
        input_scaled = scaler.transform([input_features])

        # Get the prediction from the model
        prediction = model.predict(input_scaled)[0]

        # Return the prediction, actual value as JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
