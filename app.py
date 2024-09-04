from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['blood_pressure']),
        float(data['skin_thickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetes_pedigree_function']),
        float(data['age'])
    ]
    
    # Scale the input features
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)