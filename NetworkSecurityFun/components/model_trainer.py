import os
import sys
import warnings
from typing import Any, Dict, Tuple

from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger

from NetworkSecurityFun.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from NetworkSecurityFun.entity.config_entity import ModelTrainerConfig

from NetworkSecurityFun.utils.ml_utils.model.estimator import NetworkSecurityModel
from NetworkSecurityFun.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models, 
)
from NetworkSecurityFun.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.auth.add_app_token(os.getenv("DAGSHUB_AUTH_TOKEN"))
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO"),
    mlflow=True,
)


class CyberGuardModelTrainer:
    """Train a collection of models, pick the best F‑score, track with MLflow."""

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    @staticmethod
    def calculate_haversine_distance(lat1, lng1, lat2, lng2):
        """Calculate Haversine distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lng1_rad = np.radians(lng1)
        lat2_rad = np.radians(lat2)
        lng2_rad = np.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    @staticmethod
    def calculate_geospatial_metrics(y_true, y_pred):
        """Calculate geospatial-specific metrics for withdrawal prediction"""
        # Extract coordinates (first 2 columns are lat/lng)
        true_lat, true_lng = y_true[:, 0], y_true[:, 1]
        pred_lat, pred_lng = y_pred[:, 0], y_pred[:, 1]
        
        # Calculate Haversine distance for each prediction
        distances = []
        for i in range(len(true_lat)):
            dist = CyberGuardModelTrainer.calculate_haversine_distance(
                true_lat[i], true_lng[i], pred_lat[i], pred_lng[i]
            )
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Geospatial metrics
        mean_distance_error = np.mean(distances)
        median_distance_error = np.median(distances)
        
        # Accuracy within certain radii (useful for law enforcement)
        accuracy_500m = np.mean(distances <= 0.5) * 100  # % within 500m
        accuracy_1km = np.mean(distances <= 1.0) * 100   # % within 1km
        accuracy_5km = np.mean(distances <= 5.0) * 100   # % within 5km
        
        return {
            "mean_distance_error_km": mean_distance_error,
            "median_distance_error_km": median_distance_error,
            "accuracy_within_500m_pct": accuracy_500m,
            "accuracy_within_1km_pct": accuracy_1km,
            "accuracy_within_5km_pct": accuracy_5km,
        }

    @staticmethod
    def _track_model(name: str, model: Any, r2: float, mse: float, mae: float) -> None:
        with mlflow.start_run():
            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("mean_absolute_error", mae)
            mlflow.sklearn.log_model(model, "model")

    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> ModelTrainerArtifact:

        models: Dict[str, Any] = {
            "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1)),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100)),
            "Linear Regression": MultiOutputRegressor(LinearRegression()),
            "Ridge Regression": MultiOutputRegressor(Ridge()),
            "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor()),
            "KNN": MultiOutputRegressor(KNeighborsRegressor()),
            "AdaBoost": MultiOutputRegressor(AdaBoostRegressor()),
        }

        params: Dict[str, Any] = {
            "Random Forest": {
                "estimator__n_estimators": [50, 100],
                "estimator__max_depth": [10, 20],
            },
            "Gradient Boosting": {
                "estimator__n_estimators": [50, 100],
                "estimator__learning_rate": [0.1, 0.2],
            },
            "Linear Regression": {},  # No hyperparameters to tune
            "Ridge Regression": {
                "estimator__alpha": [1.0, 10.0],
            },
            "Decision Tree": {
                "estimator__max_depth": [10, 20],
            },
            "KNN": {
                "estimator__n_neighbors": [5, 7],
            },
            "AdaBoost": {
                "estimator__n_estimators": [50, 100],
            },
        }
        scores: Dict[str, float] = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
        best_estimator = models[best_name]  

        expected = self.model_trainer_config.expected_accuracy
        if best_score < expected:
            warnings.warn(
                f"Best R² score {best_score:.3f} below expected {expected:.3f}",
                RuntimeWarning,
            )
        
        # Generate predictions
        train_pred = best_estimator.predict(X_train)
        test_pred = best_estimator.predict(X_test)
        
        # Calculate standard regression metrics
        train_r2 = r2_score(y_train, train_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Calculate geospatial metrics for withdrawal prediction
        train_geo_metrics = self.calculate_geospatial_metrics(y_train, train_pred)
        test_geo_metrics = self.calculate_geospatial_metrics(y_test, test_pred)
        
        logger.info(f"📊 Training Metrics: R²={train_r2:.4f}, Distance Error={train_geo_metrics['mean_distance_error_km']:.2f}km")
        logger.info(f"📊 Test Metrics: R²={test_r2:.4f}, Distance Error={test_geo_metrics['mean_distance_error_km']:.2f}km")
        logger.info(f"🎯 Accuracy within 1km: {test_geo_metrics['accuracy_within_1km_pct']:.1f}%")

        self._track_model(best_name, best_estimator, test_r2, test_mse, test_mae)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        full_pipeline = NetworkSecurityModel(preprocessor=preprocessor, model=best_estimator)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(self.model_trainer_config.trained_model_file_path, full_pipeline)
        save_object("final_models/model.pkl", best_estimator)

        artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            trained_metric_artifact={
                "regression_metrics": {"r2": train_r2, "mse": train_mse, "mae": train_mae},
                "geospatial_metrics": train_geo_metrics
            },
            test_metric_artifact={
                "regression_metrics": {"r2": test_r2, "mse": test_mse, "mae": test_mae},
                "geospatial_metrics": test_geo_metrics
            },
            best_model_name=best_name,
            best_model_params=best_estimator.get_params(),
        )
        logger.info(f"🏆  Best CyberGuard model: {best_name} – R²={best_score:.4f}, 1km accuracy={test_geo_metrics['accuracy_within_1km_pct']:.1f}%")
        return artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Multi-output target extraction (last 3 columns are targets)
            X_train, y_train = train_arr[:, :-3], train_arr[:, -3:]
            X_test, y_test = test_arr[:, :-3], test_arr[:, -3:]

            logger.info(f"🎯 Training data: {X_train.shape} features, {y_train.shape} targets")
            logger.info(f"🎯 Test data: {X_test.shape} features, {y_test.shape} targets")

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
