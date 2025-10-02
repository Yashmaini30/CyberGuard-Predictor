from NetworkSecurityFun.pipeline.training_pipeline import CyberGuardTrainingPipeline
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

import sys

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting CyberGuard Predictor Training Pipeline...")
        
        # Initialize CyberGuard training pipeline
        training_pipeline = CyberGuardTrainingPipeline()
        
        # Start data ingestion
        logger.info("📥 Starting data ingestion...")
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        logger.info("✅ Data ingestion completed successfully")
        
        # Start data validation  
        logger.info("🔍 Starting data validation...")
        data_validation_artifact = training_pipeline.start_data_validation(data_ingestion_artifact)
        logger.info("✅ Data validation completed successfully")

        # Start data transformation
        logger.info("🔄 Starting data transformation...")
        data_transformation_artifact = training_pipeline.start_data_transformation(data_validation_artifact)
        logger.info("✅ Data transformation completed successfully")
        
        # Start model training
        logger.info("🤖 Starting model training...")
        model_trainer_artifact = training_pipeline.start_model_trainer(data_transformation_artifact)
        logger.info("✅ Model training completed successfully")
        
        # Sync artifacts to S3 (if configured)
        logger.info("☁️ Syncing artifacts to S3...")
        training_pipeline.sync_artifact_dir_to_s3()
        logger.info("✅ S3 sync completed successfully")
        
        logger.info("🎉 CyberGuard Predictor training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        raise NetworkSecurityException(e, sys)