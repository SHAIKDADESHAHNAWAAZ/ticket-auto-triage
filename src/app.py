from prediction_api import app
import joblib

def main():
    print("Starting Ticket Auto-Triage System...")
    print("Loading model...")
    
    try:
        # Load your model here
        # model = joblib.load('models/ticket_classifier.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting API without model...")
    
    print("Starting Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
