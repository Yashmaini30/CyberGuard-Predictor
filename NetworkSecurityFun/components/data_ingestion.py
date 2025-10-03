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
            # Load from our enhanced Indian cybercrime dataset
            csv_file_path = "enhanced_indian_cybercrime_data.csv"
            logger.info(f"Loading enhanced cybercrime data from: {csv_file_path}")
            
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"Enhanced dataset file not found: {csv_file_path}")
                
            df = pd.read_csv(csv_file_path)
            logger.info(f"✅ Loaded {len(df)} enhanced cybercrime records")
            
            # Data preprocessing for withdrawal prediction
            df = self.preprocess_withdrawal_data(df)
            
            return df
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def preprocess_withdrawal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess enhanced cybercrime data for ML training"""
        try:
            logger.info("🔄 Preprocessing enhanced cybercrime data...")
            
            # Handle missing values
            df = df.dropna()
            
            # Core columns from enhanced dataset
            core_columns = [
                'complaint_id', 'timestamp', 'crime_type', 'amount_lost', 'urgency_level',
                'complaint_city', 'complaint_state', 'complaint_lat', 'complaint_lng', 'victim_phone',
                'predicted_withdrawal_lat', 'predicted_withdrawal_lng', 'withdrawal_probability', 
                'risk_score', 'status', 'reported_by', 'investigation_officer', 'bank_involved'
            ]
            
            # Keep only available columns
            available_columns = [col for col in core_columns if col in df.columns]
            df = df[available_columns]
            
            # Feature engineering for ML
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_peak_hours'] = df['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
            
            # Calculate distance if coordinates available
            if all(col in df.columns for col in ['complaint_lat', 'complaint_lng', 'predicted_withdrawal_lat', 'predicted_withdrawal_lng']):
                from math import radians, cos, sin, asin, sqrt
                def haversine(lon1, lat1, lon2, lat2):
                    # Convert to radians
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    r = 6371  # Radius of earth in kilometers
                    return c * r
                
                df['distance_km'] = df.apply(lambda row: haversine(
                    row['complaint_lng'], row['complaint_lat'],
                    row['predicted_withdrawal_lng'], row['predicted_withdrawal_lat']
                ), axis=1)
            
            # Create alert priority based on risk score and amount
            if 'risk_score' in df.columns and 'amount_lost' in df.columns:
                df['alert_priority'] = 'Low'
                df.loc[(df['risk_score'] >= 80) | (df['amount_lost'] >= 100000), 'alert_priority'] = 'High'
                df.loc[(df['risk_score'] >= 60) & (df['risk_score'] < 80) & (df['amount_lost'] >= 50000) & (df['amount_lost'] < 100000), 'alert_priority'] = 'Medium'
            
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
        


