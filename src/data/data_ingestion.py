import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def load_data(self):
        """Load data from a CSV file and store it in the feature store."""
        try:
            logging.info("Starting data ingestion process.")

            # FIX: Use correct path
            raw_data_path = os.path.join("notebooks", "data.csv")
            dataframe = pd.read_csv(raw_data_path)
            logging.info(f"Read data.csv successfully with shape {dataframe.shape}")

            # FIX: Save to feature store path
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Saved raw data to: {self.data_ingestion_config.feature_store_file_path}")

            return dataframe
        except Exception as e:
            logging.error('Unexpected error occurred while loading the data: %s', e)
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Best practice: Log after saving
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            logging.info(f"Train set saved to: {self.data_ingestion_config.training_file_path}")

            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Test set saved to: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.load_data()
            logging.info("Loaded raw data")

            self.split_data_as_train_test(dataframe)
            logging.info("Split data into train and test")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)
        
if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig
    ingestion = DataIngestion(DataIngestionConfig())
    ingestion.initiate_data_ingestion()



# import os
# import sys
# import pandas as pd
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
# from pandas import DataFrame
# from sklearn.model_selection import train_test_split

# from src.entity.config_entity import DataIngestionConfig
# from src.entity.artifact_entity import DataIngestionArtifact
# from src.exception import MyException
# from src.logger import logging

# class DataIngestion:
#     def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
#         """
#         :param data_ingestion_config: configuration for data ingestion
#         """
#         try:
#             self.data_ingestion_config = data_ingestion_config
#         except Exception as e:
#             raise MyException(e,sys)
        

#     def load_data(self):
#         """Load data from a CSV file."""
#         try:
#             logging.info("Starting data ingestion process.")

#             # Read the raw data
#             dataframe = pd.read_csv(r"notebooks\data.csv")
#             logging.info(f"Read data.csv successfully with shape {dataframe.shape}")
#             return dataframe
#         except Exception as e:
#             logging.error('Unexpected error occurred while loading the data: %s', e)
#             raise

#     def split_data_as_train_test(self,dataframe: DataFrame) ->None:
#         """
#         Method Name :   split_data_as_train_test
#         Description :   This method splits the dataframe into train set and test set based on split ratio 
        
#         Output      :   Folder is created in s3 bucket
#         On Failure  :   Write an exception log and then raise an exception
#         """
#         logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

#         try:
#             train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
#             logging.info("Performed train test split on the dataframe")
#             logging.info(
#                 "Exited split_data_as_train_test method of Data_Ingestion class"
#             )
#             dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
#             os.makedirs(dir_path,exist_ok=True)
            
#             logging.info(f"Exporting train and test file path.")
#             train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
#             test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

#             logging.info(f"Exported train and test file path.")
#         except Exception as e:
#             raise MyException(e, sys) from e

#     def initiate_data_ingestion(self) ->DataIngestionArtifact:
#         """
#         Method Name :   initiate_data_ingestion
#         Description :   This method initiates the data ingestion components of training pipeline 
        
#         Output      :   train set and test set are returned as the artifacts of data ingestion components
#         On Failure  :   Write an exception log and then raise an exception
#         """
#         logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

#         try:
#             dataframe = self.load_data()

#             logging.info("Got the data")

#             self.split_data_as_train_test(dataframe)

#             logging.info("Performed train test split on the dataset")

#             logging.info(
#                 "Exited initiate_data_ingestion method of Data_Ingestion class"
#             )

#             data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
#             test_file_path=self.data_ingestion_config.testing_file_path)
            
#             logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
#             return data_ingestion_artifact
#         except Exception as e:
#             raise MyException(e, sys) from e
