# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
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

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()