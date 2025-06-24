import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime
import yaml
import mlflow 

# TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Load the timestamp from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

TIMESTAMP: str = params["timestamp"]


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    # artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    artifact_dir: str = ARTIFACT_DIR  # NO timestamp here!
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)

    # data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    
    # transformed_train_feature_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "X_train.npy")
    # transformed_train_target_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "y_train.npy")
    
    # transformed_test_feature_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "X_test.npy")
    # transformed_test_target_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, "y_test.npy")
    
    # transformed_object_file_path: str = os.path.join(data_transformation_dir,
    #                                                  DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    #                                                  PREPROCSSING_OBJECT_FILE_NAME)
    
    transformation_status_file: str = os.path.join(data_transformation_dir, "status.txt")
    

    
@dataclass
class ModelTrainerConfig:

    # While using Constants variables for or hyperparameters values we use this code
    # model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    # trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    # expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    # model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    # _n_estimators = MODEL_TRAINER_N_ESTIMATORS
    # _min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
    # _min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
    # _max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
    # _criterion = MIN_SAMPLES_SPLIT_CRITERION
    # _random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE

    # While using Params.yaml for or hyperparameters values we use this code 
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_accuracy: float = params["model_trainer"]["expected_score"]
    model_config_file_path: str = params["model_trainer"]["model_config_file_path"]

    n_estimators: int = params["model_trainer"]["n_estimators"]
    min_samples_split: int = params["model_trainer"]["min_samples_split"]
    min_samples_leaf: int = params["model_trainer"]["min_samples_leaf"]
    max_depth: int = params["model_trainer"]["max_depth"]
    criterion: str = params["model_trainer"]["criterion"]
    random_state: int = params["model_trainer"]["random_state"]

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    pusher_model_path: str = os.path.join("saved_models", "model.pkl")  # Added to save locally for DVC

# Model registration-specific configuration
@dataclass
class ModelRegistrationConfig:
    tracking_uri: str = "https://dagshub.com/Arsalan3043/Predictive-Maintenance-Capstone-proj.mlflow"
    dagshub_repo_owner: str = "Arsalan3043"
    dagshub_repo_name: str = "Predictive-Maintenance-Capstone-proj"
    model_name: str = "my_model"
    model_info_path: str = os.path.join("reports", "experiment_info.json")  # where model run_id & model_path is stored