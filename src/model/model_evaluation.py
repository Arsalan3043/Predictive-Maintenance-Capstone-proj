import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
import dagshub

# ------------------- DagsHub & MLflow Setup ---------------------
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"

# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# mlflow.set_experiment("my-dvc-pipeline")
# ---------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/Arsalan3043/Predictive-Maintenance-Capstone-proj.mlflow')
dagshub.init(repo_owner='Arsalan3043', repo_name='Predictive-Maintenance-Capstone-proj', mlflow=True)
mlflow.set_experiment("my-dvc-pipeline")
# -------------------------------------------------------------------------------------

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)
            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain-specific feature engineering as explored in EDA.
        """
        logging.info("Applying feature engineering transformations")
        try:
            # 1. Temperature difference
            df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

            # 2. Torque per RPM (rotational speed)
            df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)

            # 3. Binary high wear flag
            critical_threshold = 202.4
            df['is_high_wear'] = (df['Tool wear [min]'] > critical_threshold).astype(int)

            # 4. Temperature-wear interaction
            df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']

            logging.info("Feature engineering completed")
            return df
        except Exception as e:
            logging.error("Feature engineering failed")
            raise MyException(e, sys)

    def _drop_id_column(self, df):
        """Drop the specified columns from schema_config if they exist."""
        drop_cols = self._schema_config['drop_columns']
        logging.info(f"Dropping columns (if exist): {drop_cols}")
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._create_dummy_columns(x)
            x = self._apply_feature_engineering(x)
            x = self._drop_id_column(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            with mlflow.start_run() as run:
                y_pred = trained_model.predict(x)
                y_proba = trained_model.predict_proba(x)[:, 1] if hasattr(trained_model, "predict_proba") else None

                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred)
                rec = recall_score(y, y_pred)
                auc = roc_auc_score(y, y_proba) if y_proba is not None else None

                metrics_dict = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'auc': auc,
                    'f1_score': trained_model_f1_score
                }

                for metric_name, metric_value in metrics_dict.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, metric_value)

                if hasattr(trained_model, "get_params"):
                    for k, v in trained_model.get_params().items():
                        mlflow.log_param(k, v)

                mlflow.sklearn.log_model(trained_model, artifact_path="model")

                save_metrics(metrics_dict, 'reports/evaluation.json')         # Changed the name from metrics.json to evaluation.json
                save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
                mlflow.log_artifact('reports/evaluation.json')                # Changed the name from metrics.json to evaluation.json

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

# Main block for DVC pipeline execution

if __name__ == "__main__":
    from src.entity.config_entity import ModelEvaluationConfig
    from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
    import json

    try:
        print("-------------------------------------------------------------")
        print("Running Model Evaluation Component via DVC pipeline")

        # Load metric values from metrics.json (NOT using dill)
        with open("artifacts/model_trainer/trained_model/metrics.json", "r") as f:
            metrics = json.load(f)

        metric_artifact = ClassificationMetricArtifact(
            f1_score=metrics["f1_score"],
            precision_score=metrics["precision"],
            recall_score=metrics["recall"]
        )

        # Construct the artifacts
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path="artifacts/model_trainer/trained_model/model.pkl",
            metric_artifact=metric_artifact
        )

        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path="artifacts/data_ingestion/ingested/train.csv",
            test_file_path="artifacts/data_ingestion/ingested/test.csv"
        )

        model_eval_config = ModelEvaluationConfig()

        model_evaluation = ModelEvaluation(
            model_eval_config=model_eval_config,
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact
        )

        # Start evaluation
        model_eval_artifact = model_evaluation.initiate_model_evaluation()
        print("Model evaluation completed.")
        print(f"Is model accepted? {model_eval_artifact.is_model_accepted}")
        print(f"Accuracy Change: {model_eval_artifact.changed_accuracy}")

    except Exception as e:
        raise MyException(e, sys)

