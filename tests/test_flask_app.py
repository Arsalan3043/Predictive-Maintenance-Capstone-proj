import unittest
from flask_app.app import app

class PredictiveMaintenanceAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page_loads(self):
        """Test if the home page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Predictive Maintenance</title>', response.data)

    def test_prediction_with_valid_input(self):
        """Test prediction route with valid input form data."""
        response = self.client.post('/predict', data={
            'Type': 'M',
            'Air temperature [K]': '298.1',
            'Process temperature [K]': '308.6',
            'Rotational speed [rpm]': '1551',
            'Torque [Nm]': '42.8',
            'Tool wear [min]': '0'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Failure Status:', response.data)
        self.assertIn(b'Failure Probability:', response.data)

    def test_metrics_endpoint(self):
        """Test /metrics endpoint for Prometheus exposure."""
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'app_request_count', response.data)
        self.assertIn(b'app_latency_seconds', response.data)

if __name__ == '__main__':
    unittest.main()
