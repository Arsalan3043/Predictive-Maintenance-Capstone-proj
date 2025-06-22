import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ------------------------------- FLASK APP INIT -------------------------------
app = Flask(__name__)

# ------------------- DagsHub & MLflow Setup ---------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Arsalan3043"
repo_name = "Predictive-Maintenance-Capstone-proj"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("my-dvc-pipeline")
# ---------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Arsalan3043/Predictive-Maintenance-Capstone-proj.mlflow')
# dagshub.init(repo_owner='Arsalan3043', repo_name='Predictive-Maintenance-Capstone-proj', mlflow=True)
# mlflow.set_experiment("my-dvc-pipeline")
# -------------------------------------------------------------------------------------

# ------------------------------- PROMETHEUS METRICS ----------------------------
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_latency_seconds", "Request latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("predicted_classes", "Predicted labels", ["label"], registry=registry)

# -------------------------------- LOAD MODEL ----------------------------------
def load_production_model():
    client = mlflow.tracking.MlflowClient()
    latest_model = client.get_latest_versions(name="my_model", stages=["Staging"])
    if not latest_model:
        raise Exception("No model found in staging")
    model_uri = f"models:/my_model/{latest_model[0].version}"
    return mlflow.pyfunc.load_model(model_uri)

model = load_production_model()

# ------------------------------ SCHEMA + FEATURES ------------------------------
schema_config = read_yaml_file(SCHEMA_FILE_PATH)
drop_columns = schema_config['drop_columns']

input_features = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

# ------------------------------ TRANSFORMATION UTILS --------------------------
def create_dummy_columns(df):
    return pd.get_dummies(df, drop_first=True)

def apply_feature_engineering(df):
    df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)
    df['is_high_wear'] = (df['Tool wear [min]'] > 202.4).astype(int)
    df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']
    return df

def drop_columns_if_exist(df):
    return df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# ---------------------------------- ROUTES ------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    resp = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return resp

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        input_data = {feature: request.form[feature] for feature in input_features}
        df = pd.DataFrame([input_data])

        # Correct dtypes
        df['Air temperature [K]'] = df['Air temperature [K]'].astype(float)
        df['Process temperature [K]'] = df['Process temperature [K]'].astype(float)
        df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype(int)
        df['Torque [Nm]'] = df['Torque [Nm]'].astype(float)
        df['Tool wear [min]'] = df['Tool wear [min]'].astype(int)

        # --- ADD THIS TO FIX MISSING COLUMNS AFTER GET_DUMMIES ---
        df = create_dummy_columns(df)

        # List of expected columns for your model input
        expected_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type_L', 'Type_M']

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Optional: reorder columns alphabetically or as your model expects
        df = df.reindex(sorted(df.columns), axis=1)
        # --- END OF FIX (Need to re-train model by droping the above columnsas they are sub target)---

        # df = create_dummy_columns(df)        # Uncomment it after removing the above fix section
        df = apply_feature_engineering(df)
        df = drop_columns_if_exist(df)

        # Prediction
        pred_proba = model.predict(df)[0] if hasattr(model, "predict") else 0.0
        pred_label = int(pred_proba >= 0.5)
        PREDICTION_COUNT.labels(label=str(pred_label)).inc()

        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        # return render_template("index.html", result=f"Failure: {pred_label} | Probability: {pred_proba:.2f}")

        # <-- CHANGE HERE: pass dict to template for result -->
        return render_template(
            "index.html",
            result={
                "label": pred_label,
                "probability": f"{pred_proba * 100:.2f}"
            }
        )

    except Exception as e:
        return f"Error during prediction: {e}"

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# --------------------------- FLASK ENTRY POINT ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
