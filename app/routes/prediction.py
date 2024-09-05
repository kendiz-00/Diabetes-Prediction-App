from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from app import db, limiter
from app.models.prediction import Prediction
import joblib
import numpy as np

bp = Blueprint('prediction', __name__)

model = joblib.load('ml_model/diabetes_model.joblib')
scaler = joblib.load('ml_model/scaler.joblib')

@bp.route('/')
@login_required
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
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
    
    # Save prediction to database
    new_prediction = Prediction(
        user_id=current_user.id,
        pregnancies=data['pregnancies'],
        glucose=data['glucose'],
        blood_pressure=data['blood_pressure'],
        skin_thickness=data['skin_thickness'],
        insulin=data['insulin'],
        bmi=data['bmi'],
        diabetes_pedigree_function=data['diabetes_pedigree_function'],
        age=data['age'],
        prediction=bool(prediction[0]),
        probability=float(probability)
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability)
    })