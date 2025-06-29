import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            # min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            # mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    # ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e


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
        import logging
        drop_cols = self._schema_config['drop_columns']
        logging.info(f"Dropping columns (if exist): {drop_cols}")
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._apply_feature_engineering(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)

            input_feature_test_df = self._apply_feature_engineering(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            # Save status file for DVC
            os.makedirs(os.path.dirname(self.data_transformation_config.transformation_status_file), exist_ok=True)
            with open(self.data_transformation_config.transformation_status_file, 'w') as f:
                f.write("Transformation completed")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e

# Main block for DVC pipeline execution
if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
    from src.data.data_ingestion import DataIngestion
    from src.data.data_validation import DataValidation

    ingestion = DataIngestion(DataIngestionConfig())
    ingestion_artifact = ingestion.initiate_data_ingestion()

    validation = DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=DataValidationConfig()
    )
    validation_artifact = validation.initiate_data_validation()

    transformation = DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_transformation_config=DataTransformationConfig(),
        data_validation_artifact=validation_artifact
    )
    transformation.initiate_data_transformation()


# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
# import pandas as pd
# from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
# from src.entity.config_entity import DataTransformationConfig
# from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
# from src.exception import MyException
# from src.logger import logging
# from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


# class DataTransformation:
#     def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
#                  data_transformation_config: DataTransformationConfig,
#                  data_validation_artifact: DataValidationArtifact):
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_transformation_config = data_transformation_config
#             self.data_validation_artifact = data_validation_artifact
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise MyException(e, sys)

#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise MyException(e, sys)

#     def get_data_transformer_object(self) -> Pipeline:
#         logging.info("Entered get_data_transformer_object method of DataTransformation class")
#         try:
#             numeric_transformer = StandardScaler()
#             logging.info("Transformers Initialized: StandardScaler")

#             num_features = self._schema_config['num_features']
#             logging.info("Cols loaded from schema.")

#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("StandardScaler", numeric_transformer, num_features),
#                 ],
#                 remainder='passthrough'
#             )

#             final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
#             logging.info("Final Pipeline Ready!!")
#             logging.info("Exited get_data_transformer_object method of DataTransformation class")
#             return final_pipeline

#         except Exception as e:
#             logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
#             raise MyException(e, sys) from e

#     def _create_dummy_columns(self, df):
#         logging.info("Creating dummy variables for categorical features")
#         df = pd.get_dummies(df, drop_first=True)
#         return df

#     def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
#         logging.info("Applying feature engineering transformations")
#         try:
#             df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
#             df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)
#             critical_threshold = 202.4
#             df['is_high_wear'] = (df['Tool wear [min]'] > critical_threshold).astype(int)
#             df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']
#             logging.info("Feature engineering completed")
#             return df
#         except Exception as e:
#             logging.error("Feature engineering failed")
#             raise MyException(e, sys)

#     def _drop_id_column(self, df):
#         drop_cols = self._schema_config['drop_columns']
#         logging.info(f"Dropping columns (if exist): {drop_cols}")
#         df = df.drop(columns=[col for col in drop_cols if col in df.columns])
#         return df

#     def initiate_data_transformation(self) -> DataTransformationArtifact:
#         try:
#             logging.info("Data Transformation Started !!!")
#             if not self.data_validation_artifact.validation_status:
#                 raise Exception(self.data_validation_artifact.message)

#             train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#             test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
#             logging.info("Train-Test data loaded")

#             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]

#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]
#             logging.info("Input and Target cols defined for both train and test df.")

#             input_feature_train_df = self._apply_feature_engineering(input_feature_train_df)
#             input_feature_train_df = self._drop_id_column(input_feature_train_df)
#             input_feature_train_df = self._create_dummy_columns(input_feature_train_df)

#             input_feature_test_df = self._apply_feature_engineering(input_feature_test_df)
#             input_feature_test_df = self._drop_id_column(input_feature_test_df)
#             input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
#             logging.info("Custom transformations applied to train and test data")

#             logging.info("Starting data transformation")
#             preprocessor = self.get_data_transformer_object()
#             logging.info("Got the preprocessor object")

#             logging.info("Initializing transformation for Training-data")
#             input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
#             logging.info("Initializing transformation for Testing-data")
#             input_feature_test_arr = preprocessor.transform(input_feature_test_df)
#             logging.info("Transformation done end to end to train-test df.")

#             logging.info("Applying SMOTEENN for handling imbalanced dataset.")
#             smt = SMOTEENN(sampling_strategy="minority")
#             input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                 input_feature_train_arr, target_feature_train_df
#             )
#             input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                 input_feature_test_arr, target_feature_test_df
#             )
#             logging.info("SMOTEENN applied to train-test df.")

#             # Make sure directory exists before saving
#             os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_feature_path), exist_ok=True)

#             # Save features and target separately for train and test
#             save_numpy_array_data(self.data_transformation_config.transformed_train_feature_path, input_feature_train_final)
#             save_numpy_array_data(self.data_transformation_config.transformed_train_target_path, np.array(target_feature_train_final))

#             save_numpy_array_data(self.data_transformation_config.transformed_test_feature_path, input_feature_test_final)
#             save_numpy_array_data(self.data_transformation_config.transformed_test_target_path, np.array(target_feature_test_final))

#             save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#             logging.info("Saving transformation object and transformed files.")

#             # Save status file for DVC
#             os.makedirs(os.path.dirname(self.data_transformation_config.transformation_status_file), exist_ok=True)
#             with open(self.data_transformation_config.transformation_status_file, 'w') as f:
#                 f.write("Transformation completed")

#             logging.info("Data transformation completed successfully")
#             return DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_feature_path=self.data_transformation_config.transformed_train_feature_path,
#                 transformed_train_target_path=self.data_transformation_config.transformed_train_target_path,
#                 transformed_test_feature_path=self.data_transformation_config.transformed_test_feature_path,
#                 transformed_test_target_path=self.data_transformation_config.transformed_test_target_path
#             )

#         except Exception as e:
#             raise MyException(e, sys) from e


# # Main block for DVC pipeline execution
# if __name__ == "__main__":
#     from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
#     from src.data.data_ingestion import DataIngestion
#     from src.data.data_validation import DataValidation

#     ingestion = DataIngestion(DataIngestionConfig())
#     ingestion_artifact = ingestion.initiate_data_ingestion()

#     validation = DataValidation(
#         data_ingestion_artifact=ingestion_artifact,
#         data_validation_config=DataValidationConfig()
#     )
#     validation_artifact = validation.initiate_data_validation()

#     transformation = DataTransformation(
#         data_ingestion_artifact=ingestion_artifact,
#         data_transformation_config=DataTransformationConfig(),
#         data_validation_artifact=validation_artifact
#     )
#     transformation.initiate_data_transformation()



# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))) # to solve src import problem
# import numpy as np
# import pandas as pd
# from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.compose import ColumnTransformer

# from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
# from src.entity.config_entity import DataTransformationConfig
# from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
# from src.exception import MyException
# from src.logger import logging
# from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


# class DataTransformation:
#     def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
#                  data_transformation_config: DataTransformationConfig,
#                  data_validation_artifact: DataValidationArtifact):
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_transformation_config = data_transformation_config
#             self.data_validation_artifact = data_validation_artifact
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise MyException(e, sys)

#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise MyException(e, sys)

#     def get_data_transformer_object(self) -> Pipeline:
#         """
#         Creates and returns a data transformer object for the data, 
#         including gender mapping, dummy variable creation, column renaming,
#         feature scaling, and type adjustments.
#         """
#         logging.info("Entered get_data_transformer_object method of DataTransformation class")

#         try:
#             # Initialize transformers
#             numeric_transformer = StandardScaler()
#             # min_max_scaler = MinMaxScaler()
#             logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

#             # Load schema configurations
#             num_features = self._schema_config['num_features']
#             # mm_columns = self._schema_config['mm_columns']
#             logging.info("Cols loaded from schema.")

#             # Creating preprocessor pipeline
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("StandardScaler", numeric_transformer, num_features),
#                     # ("MinMaxScaler", min_max_scaler, mm_columns)
#                 ],
#                 remainder='passthrough'  # Leaves other columns as they are
#             )

#             # Wrapping everything in a single pipeline
#             final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
#             logging.info("Final Pipeline Ready!!")
#             logging.info("Exited get_data_transformer_object method of DataTransformation class")
#             return final_pipeline

#         except Exception as e:
#             logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
#             raise MyException(e, sys) from e


#     def _create_dummy_columns(self, df):
#         """Create dummy variables for categorical features."""
#         logging.info("Creating dummy variables for categorical features")
#         df = pd.get_dummies(df, drop_first=True)
#         return df

#     def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply domain-specific feature engineering as explored in EDA.
#         """
#         logging.info("Applying feature engineering transformations")

#         try:
#             # 1. Temperature difference
#             df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

#             # 2. Torque per RPM (rotational speed)
#             df['torque_per_rpm'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-5)

#             # 3. Binary high wear flag
#             critical_threshold = 202.4
#             df['is_high_wear'] = (df['Tool wear [min]'] > critical_threshold).astype(int)

#             # 4. Temperature-wear interaction
#             df['temp_wear_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']

#             logging.info("Feature engineering completed")
#             return df

#         except Exception as e:
#             logging.error("Feature engineering failed")
#             raise MyException(e, sys)

#     def _drop_id_column(self, df):
#         """Drop the specified columns from schema_config if they exist."""
#         import logging
#         drop_cols = self._schema_config['drop_columns']
#         logging.info(f"Dropping columns (if exist): {drop_cols}")
#         df = df.drop(columns=[col for col in drop_cols if col in df.columns])
#         return df

#     def initiate_data_transformation(self) -> DataTransformationArtifact:
#         """
#         Initiates the data transformation component for the pipeline.
#         """
#         try:
#             logging.info("Data Transformation Started !!!")
#             if not self.data_validation_artifact.validation_status:
#                 raise Exception(self.data_validation_artifact.message)

#             # Load train and test data
#             train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#             test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
#             logging.info("Train-Test data loaded")

#             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]

#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]
#             logging.info("Input and Target cols defined for both train and test df.")

#             # Apply custom transformations in specified sequence
#             input_feature_train_df = self._apply_feature_engineering(input_feature_train_df)
#             input_feature_train_df = self._drop_id_column(input_feature_train_df)
#             input_feature_train_df = self._create_dummy_columns(input_feature_train_df)

#             input_feature_test_df = self._apply_feature_engineering(input_feature_test_df)
#             input_feature_test_df = self._drop_id_column(input_feature_test_df)
#             input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
#             logging.info("Custom transformations applied to train and test data")

#             logging.info("Starting data transformation")
#             preprocessor = self.get_data_transformer_object()
#             logging.info("Got the preprocessor object")

#             logging.info("Initializing transformation for Training-data")
#             input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
#             logging.info("Initializing transformation for Testing-data")
#             input_feature_test_arr = preprocessor.transform(input_feature_test_df)
#             logging.info("Transformation done end to end to train-test df.")

#             logging.info("Applying SMOTEENN for handling imbalanced dataset.")
#             smt = SMOTEENN(sampling_strategy="minority")
#             input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                 input_feature_train_arr, target_feature_train_df
#             )
#             input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                 input_feature_test_arr, target_feature_test_df
#             )
#             logging.info("SMOTEENN applied to train-test df.")

#             train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
#             test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
#             logging.info("feature-target concatenation done for train-test df.")

#             save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#             save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
#             save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
#             logging.info("Saving transformation object and transformed files.")

#             logging.info("Data transformation completed successfully")
#             return DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#             )

#         except Exception as e:
#             raise MyException(e, sys) from e