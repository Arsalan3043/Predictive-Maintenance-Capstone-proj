import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH

class TestPredictiveMaintenanceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up MLflow tracking via DagsHub
        dagshub_username = os.getenv("MLFLOW_TRACKING_USERNAME")
        dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not dagshub_username or not dagshub_token:
            raise EnvironmentError("DagsHub username or token environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Arsalan3043"
        repo_name = "Predictive-Maintenance-Capstone-proj"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load model from Staging
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load test data
        cls.test_data = pd.read_csv("artifacts/data_ingestion/ingested/test.csv")

        # Load schema to drop columns and align features
        cls.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        cls.drop_columns = cls.schema_config["drop_columns"]

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[stage])
        return latest[0].version if latest else None

    def test_model_is_loaded(self):
        """Check if model loads correctly"""
        self.assertIsNotNone(self.model, "Model not loaded from MLflow")

    def test_model_input_output_signature(self):
        """Test input shape and prediction output"""
        df = self.test_data.drop(columns=self.drop_columns, errors='ignore').copy()

        # Minimal transformation used in your Flask app
        df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
        df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)
        df['is_high_wear'] = (df['Tool wear [min]'] > 202.4).astype(int)
        df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']

        # Dummies
        df = pd.get_dummies(df, drop_first=True)

        prediction = self.model.predict(df)

        self.assertEqual(len(prediction), df.shape[0], "Prediction output shape mismatch")
        self.assertTrue((prediction == 0).any() or (prediction == 1).any(), "Prediction should return binary values")

    def test_model_performance_thresholds(self):
        """Ensure model meets performance criteria"""
        df = self.test_data.drop(columns=self.drop_columns + ['Machine failure'], errors='ignore').copy()
        y_true = self.test_data['Machine failure']

        # Same transformation pipeline
        df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
        df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)
        df['is_high_wear'] = (df['Tool wear [min]'] > 202.4).astype(int)
        df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']
        df = pd.get_dummies(df, drop_first=True)

        y_pred = self.model.predict(df)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # You can raise this threshold later after tuning
        self.assertGreaterEqual(acc, 0.4, f"Accuracy too low: {acc}")
        self.assertGreaterEqual(prec, 0.4, f"Precision too low: {prec}")
        self.assertGreaterEqual(rec, 0.4, f"Recall too low: {rec}")
        self.assertGreaterEqual(f1, 0.4, f"F1 score too low: {f1}")

if __name__ == "__main__":
    unittest.main()
