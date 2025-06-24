# register model

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
import json
import mlflow
from src.logger import logging
import logging
import os
import dagshub
from src.entity.config_entity import ModelRegistrationConfig
from src.entity.artifact_entity import ModelRegistrationArtifact

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_username = os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not dagshub_username or not dagshub_token:
    raise EnvironmentError("DagsHub username or token environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Arsalan3043"
repo_name = "Predictive-Maintenance-Capstone-proj"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("my-dvc-pipeline")
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/Arsalan3043/Predictive-Maintenance-Capstone-proj.mlflow')
# dagshub.init(repo_owner='Arsalan3043', repo_name='Predictive-Maintenance-Capstone-proj', mlflow=True)
# -------------------------------------------------------------------------------------


# def load_model_info(file_path: str) -> dict:
#     """Load the model info from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             model_info = json.load(file)
#         logging.debug('Model info loaded from %s', file_path)
#         return model_info
#     except FileNotFoundError:
#         logging.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error occurred while loading the model info: %s', e)
#         raise

# def register_model(model_name: str, model_info: dict):
#     """Register the model to the MLflow Model Registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)
        
#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )
        
#         logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
#     except Exception as e:
#         logging.error('Error during model registration: %s', e)
#         raise

# def main():
#     try:
#         model_info_path = 'reports/experiment_info.json'
#         model_info = load_model_info(model_info_path)
        
#         model_name = "my_model"
#         register_model(model_name, model_info)
#     except Exception as e:
#         logging.error('Failed to complete the model registration process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()

class ModelRegistrar:
    def __init__(self, config: ModelRegistrationConfig):
        """
        Initializes MLflow and DagsHub tracking config.
        """
        self.config = config
        mlflow.set_tracking_uri(self.config.tracking_uri)
        dagshub.init(
            repo_owner=self.config.dagshub_repo_owner,
            repo_name=self.config.dagshub_repo_name,
            # token=os.getenv("MLFLOW_TRACKING_PASSWORD"),  # explicitly pass token
            mlflow=True
        )
        logging.info("MLflow and DagsHub tracking initialized.")

    def load_model_info(self) -> dict:
        """
        Loads model information from a JSON file.
        """
        try:
            with open(self.config.model_info_path, 'r') as file:
                model_info = json.load(file)
            logging.info(f'Model info loaded from {self.config.model_info_path}')
            return model_info
        except FileNotFoundError:
            logging.error(f'Model info file not found: {self.config.model_info_path}')
            raise
        except Exception as e:
            logging.error(f'Error loading model info: {e}')
            raise

    def initiate_model_registration(self) -> ModelRegistrationArtifact:
        """
        Registers the model with MLflow and transitions it to 'Staging' stage.
        """
        try:
            model_info = self.load_model_info()
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

            # Register the model
            model_version = mlflow.register_model(model_uri, self.config.model_name)

            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=model_version.version,
                stage="Staging"
            )

            logging.info(f"Model '{self.config.model_name}' version {model_version.version} registered and moved to Staging.")

            return ModelRegistrationArtifact(
                registered_model_name=self.config.model_name,
                model_version=str(model_version.version),
                model_stage="Staging",
                model_uri=model_uri
            )
        except Exception as e:
            logging.error(f"Model registration failed: {e}")
            raise


if __name__ == "__main__":
    import sys
    import os
    from src.entity.config_entity import ModelRegistrationConfig
    from src.entity.artifact_entity import ModelRegistrationArtifact
    from src.exception import MyException
    from src.logger import logging

    try:
        print("-------------------------------------------------------------")
        print("Starting Model Registration Component")
        logging.info("Starting Model Registration Component")

        # Load model registration config
        model_registration_config = ModelRegistrationConfig()
        logging.info(f"Loaded ModelRegistrationConfig: {model_registration_config}")

        # Create ModelRegistrar instance
        registrar = ModelRegistrar(config=model_registration_config)
        logging.info("ModelRegistrar instance created")

        # Register and transition model
        model_registration_artifact: ModelRegistrationArtifact = registrar.initiate_model_registration()
        logging.info("Model registration completed successfully")

        # Log model artifact details
        logging.info(f"Model Name: {model_registration_artifact.registered_model_name}")
        logging.info(f"Model Version: {model_registration_artifact.model_version}")
        logging.info(f"Model Stage: {model_registration_artifact.model_stage}")
        logging.info(f"Model URI: {model_registration_artifact.model_uri}")

        print("Model registration completed.")
        print(f"Model: {model_registration_artifact.registered_model_name}")
        print(f"Version: {model_registration_artifact.model_version}")
        print(f"Stage: {model_registration_artifact.model_stage}")
        print(f"URI: {model_registration_artifact.model_uri}")

    except Exception as e:
        logging.error("Error occurred during model registration", exc_info=True)
        raise MyException(e, sys) from e
