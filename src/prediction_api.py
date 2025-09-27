from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from data_preprocessing import DataPreprocessor
import time

app = Flask(__name__)

# Global variables
model = None
preprocessor = None

def load_model():
    """Load trained model and preprocessor"""
    global model, preprocessor
    
    try:
        model_data = joblib.load('models/ticket_classifier.pkl')
        model = model_data['model']
        preprocessor = model_data['vectorizer']
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "Ticket Auto-Triage API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict ticket category",
            "/health": "GET - API health check",
            "/batch_predict": "POST - Batch predictions"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ticket category for single ticket"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'subject' not in data or 'description' not in data:
            return jsonify({"error": "Missing required fields: subject and description"}), 400
        
        # Create preprocessing instance
        preprocessor = DataPreprocessor()
        
        # Prepare data
        ticket_data = pd.DataFrame([{
            'subject': data['subject'],
            'description': data['description']
        }])
        
        ticket_data = preprocessor.prepare_features(ticket_data)
        features = preprocessor.transform_features(ticket_data['cleaned_text'])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = max(probability)
        
        # Determine priority based on confidence and content
        priority = "High" if confidence < 0.7 or any(word in data['subject'].lower() for word in ['urgent', 'critical', 'broken', 'failed']) else "Medium"
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return jsonify({
            "predicted_category": prediction,
            "confidence": round(confidence, 4),
            "suggested_priority": priority,
            "processing_time_ms": round(response_time, 2),
            "all_probabilities": {
                cls: round(prob, 4) for cls, prob in zip(model.classes_, probability)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict categories for multiple tickets"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        tickets = data.get('tickets', [])
        
        if not tickets:
            return jsonify({"error": "No tickets provided"}), 400
        
        results = []
        preprocessor = DataPreprocessor()
        
        for i, ticket in enumerate(tickets):
            ticket_data = pd.DataFrame([{
                'subject': ticket.get('subject', ''),
                'description': ticket.get('description', '')
            }])
            
            ticket_data = preprocessor.prepare_features(ticket_data)
            features = preprocessor.transform_features(ticket_data['cleaned_text'])
            
            prediction = model.predict(features)[0]
            probability = max(model.predict_proba(features)[0])
            
            results.append({
                "ticket_id": ticket.get('id', i),
                "predicted_category": prediction,
                "confidence": round(probability, 4)
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "predictions": results,
            "total_processing_time_ms": round(total_time, 2),
            "average_time_per_ticket_ms": round(total_time / len(tickets), 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
