import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import math

from NetworkSecurityFun.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurityFun.constants.training_pipeline import TARGET_COLUMNS

from NetworkSecurityFun.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from NetworkSecurityFun.entity.config_entity import DataTransformationConfig
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.utils.main_utils.utils import save_numpy_array_data,save_object


class CyberGuardDataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def calculate_distance(lat1, lng1, lat2, lng2):
        """Calculate haversine distance between two points"""
        try:
            R = 6371  # Earth's radius in kilometers
            
            lat1_rad = math.radians(lat1)
            lng1_rad = math.radians(lng1)
            lat2_rad = math.radians(lat2)
            lng2_rad = math.radians(lng2)
            
            dlat = lat2_rad - lat1_rad
            dlng = lng2_rad - lng1_rad
            
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return R * c
        except Exception as e:
            return 0
    
    def engineer_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geospatial features for withdrawal prediction"""
        try:
            logger.info("🔧 Engineering geospatial features...")
            
            # Calculate distance from complaint to nearest ATM
            if all(col in df.columns for col in ['complaint_lat', 'complaint_lng', 'nearest_atm_lat', 'nearest_atm_lng']):
                df['distance_to_atm'] = df.apply(
                    lambda row: self.calculate_distance(
                        row['complaint_lat'], row['complaint_lng'],
                        row['nearest_atm_lat'], row['nearest_atm_lng']
                    ), axis=1
                )
            
            # Create risk zones based on jurisdiction instead of state_code
            if 'jurisdiction' in df.columns:
                high_risk_jurisdictions = ['mumbai', 'delhi', 'bangalore', 'hyderabad']  # Major cities with high cybercrime
                df['high_risk_zone'] = df['jurisdiction'].str.lower().isin(high_risk_jurisdictions).astype(int)
            else:
                df['high_risk_zone'] = 0  # Default value
            
            # Time-based features
            if 'hour' in df.columns:
                df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
                df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Amount risk categories
            if 'amount_lost' in df.columns:
                df['amount_risk_level'] = pd.cut(df['amount_lost'], 
                                               bins=[0, 10000, 50000, 200000, float('inf')],
                                               labels=[1, 2, 3, 4]).astype(int)
            
            logger.info("✅ Geospatial feature engineering completed")
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def get_transformer_object(cls) -> Pipeline:
        """
        It initilizes a KNNimputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the imputer object

        Args:
            cls: DataTransformation class

        Returns:
            Pipeline
        """
        logger.info(f"Entered the get_transformer_object method of DataTransformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(f"Imputer object created. Exited the get_transformer_object method of DataTransformation class")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("🚀 Starting CyberGuard data transformation process...")
        try:
            logger.info(f"📂 Reading train file: [{self.data_validation_artifact.valid_train_file_path}]")
            train_df = CyberGuardDataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = CyberGuardDataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Apply geospatial feature engineering
            train_df = self.engineer_geospatial_features(train_df)
            test_df = self.engineer_geospatial_features(test_df)

            ## transform the data - multi-output for CyberGuard
            # Remove non-numeric columns for ML processing
            target_cols = TARGET_COLUMNS
            
            # Get numeric columns only (excluding target columns)
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in target_cols]
            
            logger.info(f"Using {len(feature_cols)} numeric features for training")
            logger.info(f"Feature columns: {feature_cols}")
            
            input_feature_train_df = train_df[feature_cols]
            target_feature_train_df = train_df[target_cols]

            ## transform test data - multi-output
            input_feature_test_df = test_df[feature_cols]
            target_feature_test_df = test_df[target_cols]

            preprocessor=self.get_transformer_object()

            logger.info("🔄 Fitting preprocessor on training data...")
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_features = preprocessor_object.transform(input_feature_test_df)

            # Handle multi-output targets
            logger.info("🎯 Processing multi-output targets...")
            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]

            ## save numpy array data
            logger.info("💾 Saving transformed data arrays...")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            ## save preprocessing object
            logger.info("🔧 Saving preprocessor objects...")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_models/preprocessor.pkl", preprocessor_object)

            ## preparing artifact
            data_transforrmation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logger.info("✅ CyberGuard data transformation completed successfully")
            return data_transforrmation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)     