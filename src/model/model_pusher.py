import sys
import os
import shutil  # CHANGED: Added to copy model file locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Uploading artifacts folder to s3 bucket")
            
            logging.info("Uploading new model to S3 bucket....")

            # Upload to S3
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            
            # CHANGED: Save model locally for DVC tracking
            local_model_dir = os.path.dirname(self.model_pusher_config.pusher_model_path)
            os.makedirs(local_model_dir, exist_ok=True)
            shutil.copy(self.model_evaluation_artifact.trained_model_path,
                        self.model_pusher_config.pusher_model_path)
            logging.info(f"Copied model to {self.model_pusher_config.pusher_model_path} for DVC tracking")

            # Build artifact
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        

if __name__ == "__main__":
    import os
    import sys
    from src.entity.config_entity import ModelPusherConfig
    from src.entity.artifact_entity import ModelEvaluationArtifact
    from src.logger import logging
    from src.exception import MyException

    try:
        print("-------------------------------------------------------------")
        print("Starting Model Pusher Component")
        logging.info("Starting Model Pusher Component")

        # Create ModelEvaluationArtifact (update the paths based on your artifact structure)
        model_evaluation_artifact = ModelEvaluationArtifact(
            is_model_accepted=True,  # or False, depending on pipeline
            changed_accuracy=0.02,   # dummy value for now
            s3_model_path="model.pkl",  # update this as needed
            trained_model_path="artifacts/model_trainer/trained_model/model.pkl"
        )

        # Load config
        model_pusher_config = ModelPusherConfig()

        # Initialize and trigger pusher
        model_pusher = ModelPusher(
            model_evaluation_artifact=model_evaluation_artifact,
            model_pusher_config=model_pusher_config
        )

        model_pusher_artifact = model_pusher.initiate_model_pusher()

        print("Model Pushing completed.")
        print(f"Bucket: {model_pusher_artifact.bucket_name}")
        print(f"Path: {model_pusher_artifact.s3_model_path}")
        logging.info("Model Pusher Component completed.")

    except Exception as e:
        logging.error("Error occurred during model pushing.")
        raise MyException(e, sys)
