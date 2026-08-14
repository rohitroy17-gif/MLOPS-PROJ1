import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH

from src.entity.config_entity import DataTransformationConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):

        try:

            self.data_ingestion_artifact = data_ingestion_artifact

            self.data_transformation_config = (
                data_transformation_config
            )

            self.data_validation_artifact = (
                data_validation_artifact
            )

            # Load schema.yaml
            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

            logging.info("Schema configuration loaded successfully")

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # READ DATA
    # =========================================================

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:

        try:

            logging.info(
                f"Reading data from: {file_path}"
            )

            return pd.read_csv(file_path)

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # DATA TRANSFORMER
    # =========================================================

    def get_data_transformer_object(self) -> Pipeline:

        """
        Creates the preprocessing pipeline.

        StandardScaler:
            Age
            Vintage

        MinMaxScaler:
            Annual_Premium

        Remaining columns:
            passed through without transformation.
        """

        try:

            logging.info(
                "Creating data transformer object"
            )

            # -------------------------------------------------
            # Scalers
            # -------------------------------------------------

            numeric_transformer = StandardScaler()

            min_max_scaler = MinMaxScaler()

            # -------------------------------------------------
            # Load columns from schema
            # -------------------------------------------------

            num_features = self._schema_config[
                "num_features"
            ]

            mm_columns = self._schema_config[
                "mm_columns"
            ]

            logging.info(
                f"StandardScaler columns: {num_features}"
            )

            logging.info(
                f"MinMaxScaler columns: {mm_columns}"
            )

            # -------------------------------------------------
            # Column Transformer
            # -------------------------------------------------

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "StandardScaler",
                        numeric_transformer,
                        num_features
                    ),

                    (
                        "MinMaxScaler",
                        min_max_scaler,
                        mm_columns
                    )
                ],

                remainder="passthrough"
            )

            # -------------------------------------------------
            # Final Pipeline
            # -------------------------------------------------

            final_pipeline = Pipeline(

                steps=[

                    (
                        "Preprocessor",
                        preprocessor
                    )
                ]
            )

            logging.info(
                "Data transformer object created successfully"
            )

            return final_pipeline

        except Exception as e:

            logging.exception(
                "Error while creating data transformer"
            )

            raise MyException(e, sys) from e

    # =========================================================
    # MAP GENDER
    # =========================================================

    def _map_gender_column(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Female -> 0
        Male   -> 1
        """

        try:

            logging.info(
                "Mapping Gender column"
            )

            if "Gender" not in df.columns:

                raise ValueError(
                    "Gender column is missing from dataframe"
                )

            df["Gender"] = df["Gender"].map(
                {
                    "Female": 0,
                    "Male": 1
                }
            )

            # Check unknown values
            if df["Gender"].isnull().any():

                raise ValueError(
                    "Gender contains unknown values. "
                    "Expected 'Female' or 'Male'."
                )

            df["Gender"] = df["Gender"].astype(int)

            return df

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # DROP ID COLUMNS
    # =========================================================

    def _drop_id_column(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Drops ID columns defined inside schema.yaml.

        Recommended schema:

        drop_columns:
          - id
          - _id
        """

        try:

            logging.info(
                "Checking for ID columns to drop"
            )

            drop_columns = self._schema_config.get(
                "drop_columns",
                []
            )

            # -------------------------------------------------
            # Convert string to list
            # -------------------------------------------------

            if isinstance(drop_columns, str):

                drop_columns = [
                    drop_columns
                ]

            # -------------------------------------------------
            # Find existing columns
            # -------------------------------------------------

            existing_columns = [

                col

                for col in drop_columns

                if col in df.columns
            ]

            # -------------------------------------------------
            # Drop columns
            # -------------------------------------------------

            if existing_columns:

                df = df.drop(
                    columns=existing_columns
                )

                logging.info(
                    f"Dropped columns: {existing_columns}"
                )

            else:

                logging.info(
                    "No configured ID columns found"
                )

            return df

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # CREATE DUMMY VARIABLES
    # =========================================================

    def _create_dummy_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):

        """
        Create dummy variables.

        Train and test are forced to have
        exactly the same columns.
        """

        try:

            logging.info(
                "Creating dummy variables"
            )

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            train_df = pd.get_dummies(
                train_df,
                drop_first=True
            )

            # -------------------------------------------------
            # Test
            # -------------------------------------------------

            test_df = pd.get_dummies(
                test_df,
                drop_first=True
            )

            # -------------------------------------------------
            # Synchronize columns
            # -------------------------------------------------

            test_df = test_df.reindex(
                columns=train_df.columns,
                fill_value=0
            )

            # -------------------------------------------------
            # Also synchronize train explicitly
            # -------------------------------------------------

            train_df = train_df.reindex(
                columns=test_df.columns,
                fill_value=0
            )

            logging.info(
                "Train and test dummy columns synchronized"
            )

            return train_df, test_df

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # RENAME COLUMNS
    # =========================================================

    def _rename_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Rename dummy columns to the names expected
        by the prediction pipeline.
        """

        try:

            logging.info(
                "Renaming dummy columns"
            )

            df = df.rename(

                columns={

                    "Vehicle_Age_< 1 Year":
                        "Vehicle_Age_lt_1_Year",

                    "Vehicle_Age_> 2 Years":
                        "Vehicle_Age_gt_2_Years"
                }
            )

            # -------------------------------------------------
            # Convert dummy columns to int
            # -------------------------------------------------

            dummy_columns = [

                "Vehicle_Age_lt_1_Year",

                "Vehicle_Age_gt_2_Years",

                "Vehicle_Damage_Yes"
            ]

            for col in dummy_columns:

                if col in df.columns:

                    df[col] = df[col].astype(int)

            return df

        except Exception as e:

            raise MyException(e, sys) from e

    # =========================================================
    # PREPARE DATAFRAME
    # =========================================================

    def _prepare_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Apply all custom transformations in the
        correct order.
        """

        logging.info(
            "Preparing dataframe for transformation"
        )

        # 1. Gender mapping
        df = self._map_gender_column(df)

        # 2. Remove ID columns
        df = self._drop_id_column(df)

        return df

    # =========================================================
    # COMPLETE DATA TRANSFORMATION
    # =========================================================

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:

        try:

            logging.info(
                "========== DATA TRANSFORMATION STARTED =========="
            )

            # =================================================
            # VALIDATION CHECK
            # =================================================

            if not self.data_validation_artifact.validation_status:

                raise Exception(
                    self.data_validation_artifact.message
                )

            # =================================================
            # LOAD TRAIN DATA
            # =================================================

            train_df = self.read_data(
                file_path=
                self.data_ingestion_artifact.trained_file_path
            )

            # =================================================
            # LOAD TEST DATA
            # =================================================

            test_df = self.read_data(
                file_path=
                self.data_ingestion_artifact.test_file_path
            )

            logging.info(
                "Train and test data loaded successfully"
            )

            logging.info(
                f"Original train shape: {train_df.shape}"
            )

            logging.info(
                f"Original test shape: {test_df.shape}"
            )

            # =================================================
            # CHECK TARGET
            # =================================================

            if TARGET_COLUMN not in train_df.columns:

                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' "
                    "not found in training dataframe"
                )

            if TARGET_COLUMN not in test_df.columns:

                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' "
                    "not found in testing dataframe"
                )

            # =================================================
            # SEPARATE FEATURES AND TARGET
            # =================================================

            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[
                TARGET_COLUMN
            ]

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[
                TARGET_COLUMN
            ]

            logging.info(
                "Features and target separated"
            )

            # =================================================
            # CUSTOM TRAIN TRANSFORMATION
            # =================================================

            input_feature_train_df = (
                self._prepare_dataframe(
                    input_feature_train_df
                )
            )

            # =================================================
            # CUSTOM TEST TRANSFORMATION
            # =================================================

            input_feature_test_df = (
                self._prepare_dataframe(
                    input_feature_test_df
                )
            )

            # =================================================
            # CREATE DUMMY VARIABLES
            # =================================================

            (
                input_feature_train_df,
                input_feature_test_df
            ) = self._create_dummy_columns(

                input_feature_train_df,

                input_feature_test_df
            )

            # =================================================
            # RENAME COLUMNS
            # =================================================

            input_feature_train_df = (
                self._rename_columns(
                    input_feature_train_df
                )
            )

            input_feature_test_df = (
                self._rename_columns(
                    input_feature_test_df
                )
            )

            logging.info(
                "Custom transformations completed"
            )

            # =================================================
            # FINAL COLUMN CHECK
            # =================================================

            logging.info(
                "Final training columns:"
            )

            logging.info(
                list(input_feature_train_df.columns)
            )

            logging.info(
                "Final testing columns:"
            )

            logging.info(
                list(input_feature_test_df.columns)
            )

            # =================================================
            # MAKE SURE TRAIN/TEST MATCH
            # =================================================

            if list(
                input_feature_train_df.columns
            ) != list(
                input_feature_test_df.columns
            ):

                raise ValueError(
                    "Training and testing columns "
                    "are not identical after preprocessing"
                )

            # =================================================
            # CREATE PREPROCESSOR
            # =================================================

            preprocessor = (
                self.get_data_transformer_object()
            )

            logging.info(
                "Preprocessor created successfully"
            )

            # =================================================
            # FIT TRAIN DATA
            # =================================================

            logging.info(
                "Fitting preprocessor on training data"
            )

            input_feature_train_arr = (
                preprocessor.fit_transform(
                    input_feature_train_df
                )
            )

            # =================================================
            # TRANSFORM TEST DATA
            # =================================================

            logging.info(
                "Transforming testing data"
            )

            input_feature_test_arr = (
                preprocessor.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Preprocessing completed successfully"
            )

            # =================================================
            # SMOTEENN - TRAIN ONLY
            # =================================================

            logging.info(
                "Applying SMOTEENN to training data only"
            )

            smt = SMOTEENN(
                sampling_strategy="minority"
            )

            (
                input_feature_train_final,
                target_feature_train_final
            ) = smt.fit_resample(

                input_feature_train_arr,

                target_feature_train_df
            )

            # -------------------------------------------------
            # DO NOT RESAMPLE TEST DATA
            # -------------------------------------------------

            input_feature_test_final = (
                input_feature_test_arr
            )

            target_feature_test_final = (
                target_feature_test_df
            )

            logging.info(
                "SMOTEENN completed successfully"
            )

            # =================================================
            # CONCATENATE FEATURES + TARGET
            # =================================================

            train_arr = np.c_[

                input_feature_train_final,

                np.asarray(
                    target_feature_train_final
                )
            ]

            test_arr = np.c_[

                input_feature_test_final,

                np.asarray(
                    target_feature_test_final
                )
            ]

            logging.info(
                f"Transformed train shape: {train_arr.shape}"
            )

            logging.info(
                f"Transformed test shape: {test_arr.shape}"
            )

            # =================================================
            # SAVE PREPROCESSOR
            # =================================================

            save_object(

                self.data_transformation_config
                .transformed_object_file_path,

                preprocessor
            )

            logging.info(
                "Preprocessor saved successfully"
            )

            # =================================================
            # SAVE TRAIN ARRAY
            # =================================================

            save_numpy_array_data(

                self.data_transformation_config
                .transformed_train_file_path,

                array=train_arr
            )

            logging.info(
                "Transformed training data saved"
            )

            # =================================================
            # SAVE TEST ARRAY
            # =================================================

            save_numpy_array_data(

                self.data_transformation_config
                .transformed_test_file_path,

                array=test_arr
            )

            logging.info(
                "Transformed testing data saved"
            )

            # =================================================
            # RETURN ARTIFACT
            # =================================================

            logging.info(
                "========== DATA TRANSFORMATION COMPLETED =========="
            )

            return DataTransformationArtifact(

                transformed_object_file_path=(

                    self.data_transformation_config
                    .transformed_object_file_path
                ),

                transformed_train_file_path=(

                    self.data_transformation_config
                    .transformed_train_file_path
                ),

                transformed_test_file_path=(

                    self.data_transformation_config
                    .transformed_test_file_path
                )
            )

        except Exception as e:

            logging.exception(
                "Exception occurred during data transformation"
            )

            raise MyException(e, sys) from e