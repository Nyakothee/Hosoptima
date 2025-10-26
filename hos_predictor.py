"""
Advanced Predictor for HOS Violation Prediction System
Real-time prediction, batch processing, API integration
Production-ready with comprehensive prediction pipeline
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PredictorConfig:
    """Configuration for predictor"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # Model configuration
        self.model_path = config.get('model_path', './trained_models')
        self.model_type = config.get('model_type', 'ensemble')  # 'lstm', 'gru', 'cnn', 'ensemble'
        self.use_ensemble = config.get('use_ensemble', True)
        
        # Preprocessing
        self.preprocessor_path = config.get('preprocessor_path', './preprocessor.pkl')
        self.feature_engineer_path = config.get('feature_engineer_path', './feature_engineer.pkl')
        
        # Prediction parameters
        self.sequence_length = config.get('sequence_length', 24)
        self.prediction_threshold = config.get('prediction_threshold', 0.5)
        self.batch_size = config.get('batch_size', 32)
        
        # Real-time parameters
        self.buffer_size = config.get('buffer_size', 100)
        self.update_interval = config.get('update_interval', 60)  # seconds
        
        # Output
        self.save_predictions = config.get('save_predictions', True)
        self.prediction_output_dir = config.get('prediction_output_dir', './predictions')
        
        # Class names
        self.class_names = config.get('class_names', [
            'Driving Time Violation',
            'Break Time Violation',
            'Weekly Hours Violation',
            'Daily Hours Violation'
        ])


class ModelLoader:
    """Load trained models and preprocessing pipelines"""
    
    @staticmethod
    def load_keras_model(model_path: str):
        """Load Keras model"""
        try:
            from deep_learning_models import AttentionLayer
            
            model = keras.models.load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_ensemble_models(ensemble_dir: str) -> Dict:
        """Load ensemble of models"""
        models = {}
        ensemble_path = Path(ensemble_dir)
        
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")
        
        for model_file in ensemble_path.glob('*_model.h5'):
            model_name = model_file.stem.replace('_model', '')
            try:
                model = ModelLoader.load_keras_model(str(model_file))
                models[model_name] = model
                logger.info(f"Loaded {model_name} from ensemble")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        if not models:
            raise ValueError("No models loaded from ensemble")
        
        return models
    
    @staticmethod
    def load_preprocessor(preprocessor_path: str):
        """Load preprocessing pipeline"""
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            return preprocessor
        except Exception as e:
            logger.warning(f"Failed to load preprocessor: {str(e)}")
            return None
    
    @staticmethod
    def load_feature_engineer(feature_engineer_path: str):
        """Load feature engineering pipeline"""
        try:
            with open(feature_engineer_path, 'rb') as f:
                feature_engineer = pickle.load(f)
            logger.info(f"Loaded feature engineer from {feature_engineer_path}")
            return feature_engineer
        except Exception as e:
            logger.warning(f"Failed to load feature engineer: {str(e)}")
            return None


class InputValidator:
    """Validate input data for predictions"""
    
    @staticmethod
    def validate_input_shape(data: np.ndarray, expected_shape: Tuple) -> bool:
        """Validate input data shape"""
        if len(data.shape) != len(expected_shape):
            logger.error(f"Shape mismatch: expected {len(expected_shape)} dims, got {len(data.shape)}")
            return False
        
        for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
            if expected is not None and actual != expected:
                logger.error(f"Dimension {i} mismatch: expected {expected}, got {actual}")
                return False
        
        return True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate dataframe has required columns"""
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False, list(missing_cols)
        
        return True, []
    
    @staticmethod
    def validate_value_ranges(df: pd.DataFrame, range_dict: Dict[str, Tuple]) -> bool:
        """Validate values are within expected ranges"""
        for col, (min_val, max_val) in range_dict.items():
            if col in df.columns:
                if df[col].min() < min_val or df[col].max() > max_val:
                    logger.warning(f"Column {col} has values outside range [{min_val}, {max_val}]")
                    return False
        
        return True


class PredictionFormatter:
    """Format prediction results"""
    
    @staticmethod
    def format_binary_predictions(predictions: np.ndarray,
                                  threshold: float = 0.5,
                                  class_names: List[str] = None) -> List[Dict]:
        """Format binary classification predictions"""
        results = []
        
        for i, pred in enumerate(predictions):
            if len(pred.shape) > 0 and pred.shape[0] > 1:
                # Multi-class
                pred_class = np.argmax(pred)
                confidence = float(pred[pred_class])
            else:
                # Binary
                pred_class = 1 if pred > threshold else 0
                confidence = float(pred if pred_class == 1 else 1 - pred)
            
            result = {
                'sample_id': i,
                'predicted_class': int(pred_class),
                'confidence': confidence,
                'violation_detected': bool(pred_class > 0),
                'timestamp': datetime.now().isoformat()
            }
            
            if class_names:
                result['violation_type'] = class_names[pred_class]
            
            results.append(result)
        
        return results
    
    @staticmethod
    def format_probability_predictions(predictions: np.ndarray,
                                      class_names: List[str] = None) -> List[Dict]:
        """Format predictions with probabilities for all classes"""
        results = []
        
        for i, pred in enumerate(predictions):
            result = {
                'sample_id': i,
                'predicted_class': int(np.argmax(pred)),
                'timestamp': datetime.now().isoformat(),
                'probabilities': {}
            }
            
            for j, prob in enumerate(pred):
                class_name = class_names[j] if class_names else f'Class_{j}'
                result['probabilities'][class_name] = float(prob)
            
            # Add risk level
            max_prob = float(np.max(pred))
            if max_prob > 0.8:
                result['risk_level'] = 'HIGH'
            elif max_prob > 0.6:
                result['risk_level'] = 'MEDIUM'
            else:
                result['risk_level'] = 'LOW'
            
            results.append(result)
        
        return results
    
    @staticmethod
    def create_prediction_dataframe(predictions: List[Dict]) -> pd.DataFrame:
        """Convert predictions to DataFrame"""
        return pd.DataFrame(predictions)


class RealTimePredictor:
    """Real-time prediction with streaming data"""
    
    def __init__(self, model, config: PredictorConfig):
        self.model = model
        self.config = config
        self.buffer = deque(maxlen=config.buffer_size)
        self.sequence_buffer = deque(maxlen=config.sequence_length)
        
    def add_data_point(self, data_point: np.ndarray):
        """Add new data point to buffer"""
        self.sequence_buffer.append(data_point)
    
    def predict_current_state(self) -> Optional[Dict]:
        """Predict based on current buffer state"""
        if len(self.sequence_buffer) < self.config.sequence_length:
            logger.warning(f"Insufficient data: {len(self.sequence_buffer)}/{self.config.sequence_length}")
            return None
        
        # Create sequence
        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, self.config.sequence_length, -1)
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)
        
        # Format
        formatter = PredictionFormatter()
        result = formatter.format_probability_predictions(
            prediction,
            self.config.class_names
        )[0]
        
        return result
    
    def get_prediction_history(self, n: int = 10) -> List[Dict]:
        """Get last n predictions"""
        return list(self.buffer)[-n:]


class BatchPredictor:
    """Batch prediction for large datasets"""
    
    def __init__(self, model, config: PredictorConfig):
        self.model = model
        self.config = config
        
    def predict_batch(self, X: np.ndarray,
                     return_probabilities: bool = True) -> List[Dict]:
        """Predict on batch of data"""
        logger.info(f"Making predictions on batch of {len(X)} samples...")
        
        # Predict in batches
        predictions = self.model.predict(
            X,
            batch_size=self.config.batch_size,
            verbose=1
        )
        
        # Format predictions
        formatter = PredictionFormatter()
        
        if return_probabilities:
            results = formatter.format_probability_predictions(
                predictions,
                self.config.class_names
            )
        else:
            results = formatter.format_binary_predictions(
                predictions,
                self.config.prediction_threshold,
                self.config.class_names
            )
        
        logger.info(f"Predictions completed for {len(results)} samples")
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame,
                         feature_columns: List[str]) -> pd.DataFrame:
        """Predict on DataFrame"""
        # Extract features
        X = df[feature_columns].values
        
        # Reshape if needed
        if len(X.shape) == 2:
            # Assuming each row is a sequence
            n_samples = X.shape[0] // self.config.sequence_length
            X = X[:n_samples * self.config.sequence_length].reshape(
                n_samples, self.config.sequence_length, -1
            )
        
        # Predict
        predictions = self.predict_batch(X)
        
        # Add predictions to dataframe
        pred_df = pd.DataFrame(predictions)
        
        return pred_df


class EnsemblePredictor:
    """Prediction using ensemble of models"""
    
    def __init__(self, models: Dict, config: PredictorConfig):
        self.models = models
        self.config = config
        self.weights = self._calculate_weights()
        
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate weights for ensemble (equal by default)"""
        n_models = len(self.models)
        return {name: 1.0 / n_models for name in self.models.keys()}
    
    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for models"""
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def predict(self, X: np.ndarray, method: str = 'weighted_average') -> np.ndarray:
        """Make ensemble predictions"""
        logger.info(f"Making ensemble predictions using {method}...")
        
        all_predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            all_predictions.append(pred)
            logger.info(f"Got predictions from {model_name}")
        
        all_predictions = np.array(all_predictions)
        
        if method == 'average':
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif method == 'weighted_average':
            weights = np.array([self.weights[name] for name in self.models.keys()])
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        elif method == 'max':
            ensemble_pred = np.max(all_predictions, axis=0)
        elif method == 'voting':
            # Hard voting for classification
            votes = np.argmax(all_predictions, axis=-1)
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=votes
            )
        else:
            logger.warning(f"Unknown method {method}, using weighted average")
            weights = np.array([self.weights[name] for name in self.models.keys()])
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        
        logger.info("Ensemble predictions completed")
        
        return ensemble_pred


class AdvancedPredictor:
    """
    Comprehensive prediction system for HOS violation prediction
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()
        self.model = None
        self.models = None
        self.preprocessor = None
        self.feature_engineer = None
        self.validator = InputValidator()
        self.formatter = PredictionFormatter()
        
        # Create output directory
        Path(self.config.prediction_output_dir).mkdir(exist_ok=True, parents=True)
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load models and preprocessing pipelines"""
        logger.info("Loading prediction components...")
        
        # Load preprocessor
        if Path(self.config.preprocessor_path).exists():
            self.preprocessor = ModelLoader.load_preprocessor(self.config.preprocessor_path)
        
        # Load feature engineer
        if Path(self.config.feature_engineer_path).exists():
            self.feature_engineer = ModelLoader.load_feature_engineer(self.config.feature_engineer_path)
        
        # Load model(s)
        if self.config.use_ensemble:
            ensemble_dir = f"{self.config.model_path}/ensemble"
            if Path(ensemble_dir).exists():
                self.models = ModelLoader.load_ensemble_models(ensemble_dir)
                logger.info(f"Loaded {len(self.models)} models in ensemble")
        else:
            model_path = f"{self.config.model_path}/{self.config.model_type}_model.h5"
            if Path(model_path).exists():
                self.model = ModelLoader.load_keras_model(model_path)
    
    def preprocess_input(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Preprocess input data"""
        logger.info("Preprocessing input data...")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Apply preprocessing if available
        if self.preprocessor:
            try:
                data = self.preprocessor.transform(data)
            except Exception as e:
                logger.warning(f"Preprocessing failed: {str(e)}")
        
        # Apply feature engineering if available
        if self.feature_engineer:
            try:
                data = self.feature_engineer.transform(data)
            except Exception as e:
                logger.warning(f"Feature engineering failed: {str(e)}")
        
        return data
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                return_probabilities: bool = True,
                preprocess: bool = True) -> List[Dict]:
        """Make predictions on input data"""
        logger.info(f"Making predictions on {len(X)} samples...")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Preprocess
        if preprocess:
            X = self.preprocess_input(X)
        
        # Ensure correct shape
        if len(X.shape) == 2:
            # Reshape to sequences
            n_samples = X.shape[0] // self.config.sequence_length
            n_features = X.shape[1]
            X = X[:n_samples * self.config.sequence_length].reshape(
                n_samples, self.config.sequence_length, n_features
            )
        
        # Make predictions
        if self.config.use_ensemble and self.models:
            ensemble_predictor = EnsemblePredictor(self.models, self.config)
            predictions = ensemble_predictor.predict(X, method='weighted_average')
        elif self.model:
            predictions = self.model.predict(X, batch_size=self.config.batch_size, verbose=0)
        else:
            raise ValueError("No model loaded for predictions")
        
        # Format results
        if return_probabilities:
            results = self.formatter.format_probability_predictions(
                predictions,
                self.config.class_names
            )
        else:
            results = self.formatter.format_binary_predictions(
                predictions,
                self.config.prediction_threshold,
                self.config.class_names
            )
        
        # Save predictions if configured
        if self.config.save_predictions:
            self._save_predictions(results)
        
        logger.info(f"Predictions completed: {len(results)} results")
        
        return results
    
    def predict_single(self, data: Union[np.ndarray, Dict]) -> Dict:
        """Predict on a single sample"""
        if isinstance(data, dict):
            # Convert dict to array
            data = np.array([list(data.values())])
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        results = self.predict(data, return_probabilities=True)
        
        return results[0] if results else None
    
    def predict_driver_risk(self, driver_id: str, 
                           recent_data: pd.DataFrame) -> Dict:
        """Predict risk for specific driver"""
        logger.info(f"Predicting risk for driver {driver_id}...")
        
        # Filter data for driver
        driver_data = recent_data[recent_data['driver_id'] == driver_id]
        
        if len(driver_data) < self.config.sequence_length:
            logger.warning(f"Insufficient data for driver {driver_id}")
            return {
                'driver_id': driver_id,
                'status': 'insufficient_data',
                'data_points': len(driver_data),
                'required': self.config.sequence_length
            }
        
        # Take most recent sequence
        driver_sequence = driver_data.tail(self.config.sequence_length)
        
        # Predict
        predictions = self.predict(driver_sequence, return_probabilities=True)
        
        # Aggregate risk
        result = predictions[0] if predictions else {}
        result['driver_id'] = driver_id
        result['data_points_used'] = len(driver_sequence)
        
        return result
    
    def predict_fleet_risk(self, fleet_data: pd.DataFrame,
                          driver_col: str = 'driver_id') -> pd.DataFrame:
        """Predict risk for entire fleet"""
        logger.info("Predicting risk for fleet...")
        
        fleet_predictions = []
        
        drivers = fleet_data[driver_col].unique()
        
        for driver_id in drivers:
            driver_pred = self.predict_driver_risk(driver_id, fleet_data)
            fleet_predictions.append(driver_pred)
        
        fleet_df = pd.DataFrame(fleet_predictions)
        
        # Sort by risk level
        if 'risk_level' in fleet_df.columns:
            risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            fleet_df['risk_order'] = fleet_df['risk_level'].map(risk_order)
            fleet_df = fleet_df.sort_values('risk_order').drop('risk_order', axis=1)
        
        logger.info(f"Fleet risk assessment completed for {len(drivers)} drivers")
        
        return fleet_df
    
    def generate_alerts(self, predictions: List[Dict],
                       alert_threshold: float = 0.7) -> List[Dict]:
        """Generate alerts for high-risk predictions"""
        alerts = []
        
        for pred in predictions:
            if pred.get('risk_level') == 'HIGH' or pred.get('confidence', 0) > alert_threshold:
                alert = {
                    'alert_id': f"ALERT_{pred['sample_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'timestamp': pred['timestamp'],
                    'violation_type': pred.get('violation_type', 'Unknown'),
                    'confidence': pred.get('confidence', 0),
                    'risk_level': pred.get('risk_level', 'UNKNOWN'),
                    'action_required': True,
                    'message': f"High risk of {pred.get('violation_type', 'violation')} detected"
                }
                
                alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def _save_predictions(self, predictions: List[Dict]):
        """Save predictions to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = f"{self.config.prediction_output_dir}/predictions_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(predictions)
        csv_path = f"{self.config.prediction_output_dir}/predictions_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Predictions saved to {json_path} and {csv_path}")
    
    def get_prediction_summary(self, predictions: List[Dict]) -> Dict:
        """Generate summary statistics for predictions"""
        summary = {
            'total_predictions': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        if not predictions:
            return summary
        
        # Count by risk level
        risk_levels = [p.get('risk_level') for p in predictions if 'risk_level' in p]
        summary['risk_distribution'] = {
            'HIGH': risk_levels.count('HIGH'),
            'MEDIUM': risk_levels.count('MEDIUM'),
            'LOW': risk_levels.count('LOW')
        }
        
        # Count violations detected
        violations = [p.get('violation_detected') for p in predictions if 'violation_detected' in p]
        summary['violations_detected'] = violations.count(True)
        summary['violation_rate'] = violations.count(True) / len(violations) if violations else 0
        
        # Average confidence
        confidences = [p.get('confidence') for p in predictions if 'confidence' in p]
        summary['average_confidence'] = np.mean(confidences) if confidences else 0
        
        return summary


# Main execution
if __name__ == "__main__":
    # Example usage
    config = PredictorConfig({
        'model_path': './trained_models',
        'use_ensemble': True,
        'save_predictions': True
    })
    
    predictor = AdvancedPredictor(config)
    
    # Example: Single prediction
    # sample_data = np.random.randn(24, 50)  # 24 timesteps, 50 features
    # result = predictor.predict_single(sample_data)
    # print(json.dumps(result, indent=2))
    
    # Example: Batch prediction
    # batch_data = np.random.randn(100, 24, 50)
    # results = predictor.predict(batch_data)
    # summary = predictor.get_prediction_summary(results)
    # print(json.dumps(summary, indent=2))
    
    # Example: Generate alerts
    # alerts = predictor.generate_alerts(results, alert_threshold=0.7)
    # print(f"Generated {len(alerts)} alerts")