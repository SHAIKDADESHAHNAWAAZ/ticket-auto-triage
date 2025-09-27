import unittest
import requests
import json

class TestTicketAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5000"
    
    def test_health_check(self):
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        
    def test_single_prediction(self):
        payload = {
            "subject": "Payment failed during transaction",
            "description": "I tried to make a purchase but the payment failed with error code 500"
        }
        
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('predicted_category', data)
        self.assertIn('confidence', data)

if __name__ == '__main__':
    unittest.main()
