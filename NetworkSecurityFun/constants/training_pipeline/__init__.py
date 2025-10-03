import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for CyberGuard withdrawal prediction pipeline
"""
TARGET_COLUMNS = ["withdrawal_probability", "risk_score"]
PIPELINE_NAME: str = "cyberguard_predictor"
ARTIFACT_DIR: str = "artifact"
FILENAME: str = "datasets/withdrawal_prediction/complaint_to_withdrawal_focused.csv"

TRAIN_FILENAME: str = "train.csv"
TEST_FILENAME: str = "test.csv"

SCHEMA_FILE_PATH: str = os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR: str = os.path.join("saved_models")
MODEL_FILE_NAME: str = "model.pkl"

"""
Data Ingestion related constant starts with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "EnhancedCybercrimeData"
DATA_INGESTION_DATABASE_NAME: str = "CyberGuardDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation related constant starts with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Transformation related constant starts with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_NAME: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Multi-target transformation parameters
DATA_TRANSFORMATION_MULTI_OUTPUT_TARGETS: list = ["predicted_withdrawal_lat", "predicted_withdrawal_lng", "withdrawal_probability"]
DATA_TRANSFORMATION_GEOSPATIAL_FEATURES: list = ["complaint_lat", "complaint_lng", "nearest_atm_lat", "nearest_atm_lng"]
DATA_TRANSFORMATION_TEMPORAL_FEATURES: list = ["hour", "day_of_week", "is_weekend", "is_peak_withdrawal_time"]

# KNN Imputer parameters for withdrawal prediction
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "n_neighbors": 5,
    "weights": "uniform"
}

## knn imputerto replace missing values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 5,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test.npy"

"""
Model Trainer related constant starts with MODEL_TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.9
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05