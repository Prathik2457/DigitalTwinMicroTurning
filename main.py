from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the lightweight XGBoost model and scaler
model = joblib.load('machining_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Arrange inputs
        raw_inputs = pd.DataFrame([[
            data['speed'], 
            data['feed'], 
            data['depth'], 
            data['time']
        ]])
        
        # 2. Scale inputs
        scaled_inputs = scaler.transform(raw_inputs)
        
        # 3. Predict
        prediction = model.predict(scaled_inputs)
        
        return jsonify({
            "ra": float(round(prediction[0][0], 3)),
            "wear": float(round(prediction[0][1], 1))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()
