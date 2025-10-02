from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

## configuration of data ingestion config
from NetworkSecurityFun.entity.config_entity import DataIngestionConfig
from NetworkSecurityFun.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class CyberGuardDataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("🔄 CyberGuard Data Ingestion initialized for withdrawal prediction")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def load_withdrawal_prediction_data_from_csv(self):
        """Load withdrawal prediction data from CSV file"""
        try:
            # Load from our focused dataset
            csv_file_path = "datasets/withdrawal_prediction/complaint_to_withdrawal_focused.csv"
            logger.info(f"Loading withdrawal prediction data from: {csv_file_path}")
            
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"Dataset file not found: {csv_file_path}")
                
            df = pd.read_csv(csv_file_path)
            logger.info(f"✅ Loaded {len(df)} withdrawal prediction records")
            
            # Data preprocessing for withdrawal prediction
            df = self.preprocess_withdrawal_data(df)
            
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def preprocess_withdrawal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess withdrawal prediction data while maintaining schema compatibility"""
        try:
            logger.info("🔄 Preprocessing withdrawal prediction data...")
            
            # Handle missing values
            df = df.dropna()
            
            # Ensure we have all required columns from schema
            required_columns = [
                'complaint_id', 'timestamp', 'crime_type', 'amount_lost', 'urgency_level',
                'complaint_city', 'complaint_lat', 'complaint_lng', 'predicted_withdrawal_lat',
                'predicted_withdrawal_lng', 'withdrawal_probability', 'hours_to_withdrawal',
                'intervention_window_hours', 'hour', 'day_of_week', 'is_weekend',
                'is_peak_withdrawal_time', 'nearest_atm_network', 'nearest_atm_lat',
                'nearest_atm_lng', 'atm_distance_km', 'risk_score', 'alert_priority',
                'requires_bank_alert', 'jurisdiction'
            ]
            
            # Keep only schema-required columns
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
            
            # Add missing columns with default values if needed
            for col in required_columns:
                if col not in df.columns:
                    if col == 'complaint_id':
                        df[col] = range(len(df))
                    elif col == 'timestamp':
                        df[col] = pd.Timestamp.now()
                    elif col in ['crime_type', 'urgency_level', 'complaint_city', 'nearest_atm_network', 'alert_priority', 'jurisdiction']:
                        df[col] = 'unknown'
                    elif col == 'intervention_window_hours':
                        df[col] = 24  # Default intervention window
                    else:
                        df[col] = 0
            
            # Ensure column order matches schema
            df = df[required_columns]
            
            logger.info(f"✅ Preprocessing complete. Final shape: {df.shape}")
            logger.info(f"Final columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self):
        """Export withdrawal prediction data as dataframe"""
        try:
            # Use CSV data instead of MongoDB for now
            logger.info("📊 Loading withdrawal prediction data from CSV...")
            df = self.load_withdrawal_prediction_data_from_csv()
            
            # Optionally sync to MongoDB for future use
            if MONGO_DB_URL:
                self.sync_to_mongodb(df)
            
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def sync_to_mongodb(self, df: pd.DataFrame):
        """Sync data to MongoDB for future use"""
        try:
            logger.info("🔄 Syncing data to MongoDB...")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            
            # Convert DataFrame to records and insert
            records = df.to_dict('records')
            collection.delete_many({})  # Clear existing data
            collection.insert_many(records)
            
            logger.info(f"✅ Synced {len(records)} records to MongoDB")
            
        except Exception as e:
            logger.warning(f"MongoDB sync failed: {e}. Continuing with CSV data...")
            pass
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            ## creaing folder
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logger.info(f"Splitting of data into train and test is completed")
            logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logger.info(f"Exporting train and test file path")
            train_set.to_csv(
                self.data_ingestion_config.train_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.test_file_path, index=False, header=True
            )
            logger.info("Exporting of train and test file completed")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logger.info("🚀 Starting CyberGuard data ingestion for withdrawal prediction...")
            
            # Load withdrawal prediction data from CSV
            dataframe = self.load_withdrawal_prediction_data_from_csv()
            
            # Optional: Sync to MongoDB for future use
            self.sync_to_mongodb(dataframe)
            
            # Export to feature store
            dataframe = self.export_data_into_feature_store(dataframe)
            
            # Split for training and testing
            self.split_data_as_train_test(dataframe)
            
            # Create data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logger.info("✅ CyberGuard data ingestion completed successfully")
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        


