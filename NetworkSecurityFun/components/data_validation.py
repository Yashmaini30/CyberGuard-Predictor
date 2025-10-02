from NetworkSecurityFun.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from NetworkSecurityFun.entity.config_entity import DataValidationConfig
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from NetworkSecurityFun.utils.main_utils.utils import read_yaml_file, write_yaml_file

class CyberGuardDataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info("🔄 CyberGuard Data Validation initialized for withdrawal prediction")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logger.info(f"Required number of columns: {number_of_columns}")
            logger.info(f"Actual number of columns: {len(dataframe.columns)}")
            logger.info(f"Dataframe columns: {list(dataframe.columns)}")
            
            # Get schema column names
            schema_columns = [list(col.keys())[0] for col in self.schema_config["columns"]]
            logger.info(f"Schema columns: {schema_columns}")
            
            # Check if all schema columns exist in dataframe
            missing_cols = set(schema_columns) - set(dataframe.columns)
            extra_cols = set(dataframe.columns) - set(schema_columns)
            
            if missing_cols:
                logger.error(f"❌ Missing columns in dataset: {missing_cols}")
                return False
            if extra_cols:
                logger.error(f"❌ Extra columns in dataset: {extra_cols}")
                return False
                
            logger.info("✅ Column validation passed")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_withdrawal_prediction_data(self, dataframe: pd.DataFrame) -> bool:
        """Validate withdrawal prediction specific requirements"""
        try:
            logger.info("🔍 Validating withdrawal prediction data requirements...")
            
            # Check for required target columns
            required_targets = ["predicted_withdrawal_lat", "predicted_withdrawal_lng", "withdrawal_probability"]
            missing_targets = [col for col in required_targets if col not in dataframe.columns]
            
            if missing_targets:
                logger.error(f"❌ Missing target columns: {missing_targets}")
                return False
            
            # Validate coordinate ranges (India bounds)
            if not self.validate_indian_coordinates(dataframe):
                return False
                
            # Validate withdrawal probability range
            if not dataframe['withdrawal_probability'].between(0, 1).all():
                logger.error("❌ Withdrawal probability values outside [0,1] range")
                return False
                
            # Validate positive amounts
            if (dataframe['amount_lost'] <= 0).any():
                logger.error("❌ Found non-positive amount values")
                return False
                
            logger.info("✅ Withdrawal prediction data validation passed")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_indian_coordinates(self, dataframe: pd.DataFrame) -> bool:
        """Validate that coordinates are within Indian geographic bounds"""
        try:
            # India bounds: Latitude: 6.0° to 37.6°, Longitude: 67.7° to 97.25°
            lat_cols = ['complaint_lat', 'predicted_withdrawal_lat', 'nearest_atm_lat']
            lng_cols = ['complaint_lng', 'predicted_withdrawal_lng', 'nearest_atm_lng']
            
            for col in lat_cols:
                if col in dataframe.columns:
                    if not dataframe[col].between(6.0, 37.6).all():
                        logger.error(f"❌ Latitude values in {col} outside Indian bounds")
                        return False
                        
            for col in lng_cols:
                if col in dataframe.columns:
                    if not dataframe[col].between(67.7, 97.25).all():
                        logger.error(f"❌ Longitude values in {col} outside Indian bounds")
                        return False
                        
            logger.info("✅ Geographic coordinates validation passed")
            return True
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """Detect drift between base and current datasets for withdrawal prediction"""
        try:
            logger.info("🔍 Detecting dataset drift for withdrawal prediction...")
            
            # For withdrawal prediction, we'll focus on key numerical features
            numerical_cols = ['amount_lost', 'complaint_lat', 'complaint_lng', 
                            'withdrawal_probability', 'hours_to_withdrawal', 
                            'atm_distance_km', 'risk_score']
            
            drift_detected = False
            drift_report = {}
            
            for col in numerical_cols:
                if col in base_df.columns and col in current_df.columns:
                    # Simple statistical drift detection using mean difference
                    base_mean = base_df[col].mean()
                    current_mean = current_df[col].mean()
                    
                    if base_mean != 0:
                        drift_ratio = abs(current_mean - base_mean) / abs(base_mean)
                        drift_report[col] = {
                            'base_mean': base_mean,
                            'current_mean': current_mean,
                            'drift_ratio': drift_ratio,
                            'drift_detected': drift_ratio > threshold
                        }
                        
                        if drift_ratio > threshold:
                            drift_detected = True
                            logger.warning(f"⚠️ Drift detected in {col}: {drift_ratio:.4f}")
            
            # Save drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save report as YAML
            from NetworkSecurityFun.utils.main_utils.utils import write_yaml_file
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)
            
            if drift_detected:
                logger.warning("⚠️ Dataset drift detected - review required")
            else:
                logger.info("✅ No significant dataset drift detected")
                
            return not drift_detected  # Return True if no drift
            
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}. Continuing...")
            return True  # Continue even if drift detection fails
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                ks_2samp_result = ks_2samp(d1, d2)

                if ks_2samp_result.pvalue < threshold:
                    is_drifted = True
                    status = False
                else:
                    is_drifted = False

                report[column] = {
                    "p_value": float(ks_2samp_result.pvalue),
                    "drift_status": is_drifted
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status  # Return the drift check result
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read train and test data
            train_df = CyberGuardDataValidation.read_data(train_file_path)
            test_df = CyberGuardDataValidation.read_data(test_file_path)

            # Validate number of columns
            if not self.validate_number_of_columns(train_df):
                raise ValueError("Train dataset column count does not match schema.")
            if not self.validate_number_of_columns(test_df):
                raise ValueError("Test dataset column count does not match schema.")

            # Detect dataset drift
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact 

        except Exception as e:
            raise NetworkSecurityException(e, sys)
