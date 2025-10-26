"""
Advanced Model Trainer for HOS Violation Prediction System
Handles training, hyperparameter tuning, cross-validation, and model selection
Production-ready with comprehensive training pipeline
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow import keras
import optuna
from optuna.integration import TFKerasPruningCallback
import json
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from deep_learning_models import (
    LSTMModel, GRUModel, CNNModel, TransformerModel,
    HybridModel, EnsembleModel, ModelConfig
)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for model training"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # Data split
        self.test_size = config.get('test_size', 0.2)
        self.validation_size = config.get('validation_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Cross-validation
        self.use_cv = config.get('use_cv', True)
        self.cv_folds = config.get('cv_folds', 5)
        self.cv_type = config.get('cv_type', 'stratified')  # 'stratified' or 'timeseries'
        
        # Hyperparameter tuning
        self.use_hyperparameter_tuning = config.get('use_hyperparameter_tuning', True)
        self.tuning_method = config.get('tuning_method', 'optuna')  # 'optuna', 'grid', 'random'
        self.n_trials = config.get('n_trials', 100)
        self.timeout = config.get('timeout', 3600)  # 1 hour
        
        # Class imbalance handling
        self.use_class_weights = config.get('use_class_weights', True)
        self.use_smote = config.get('use_smote', False)
        
        # Model selection
        self.models_to_train = config.get('models_to_train', ['lstm', 'gru', 'cnn'])
        self.ensemble_models = config.get('ensemble_models', True)
        
        # Training parameters
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.reduce_lr_patience = config.get('reduce_lr_patience', 5)
        
        # Saving
        self.save_best_only = config.get('save_best_only', True)
        self.model_save_dir = config.get('model_save_dir', './trained_models')


class DataPreparator:
    """Prepare data for training"""
    
    @staticmethod
    def create_sequences(data: np.ndarray, 
                        labels: np.ndarray,
                        sequence_length: int = 24,
                        overlap: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data"""
        X, y = [], []
        
        step = sequence_length - overlap
        
        for i in range(0, len(data) - sequence_length + 1, step):
            X.append(data[i:i + sequence_length])
            # Take the last label in the sequence
            y.append(labels[i + sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        
        return X, y
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray,
                  test_size: float = 0.2,
                  validation_size: float = 0.2,
                  random_state: int = 42,
                  stratify: bool = True) -> Tuple:
        """Split data into train, validation, and test sets"""
        
        stratify_param = y if stratify else None
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Second split: train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_param = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get unique classes
        if len(y.shape) > 1:
            # One-hot encoded
            classes = np.argmax(y, axis=1)
            unique_classes = np.unique(classes)
        else:
            classes = y
            unique_classes = np.unique(y)
        
        # Compute weights
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=classes
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        logger.info(f"Calculated class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    @staticmethod
    def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling imbalanced data"""
        from imblearn.over_sampling import SMOTE
        
        # Reshape for SMOTE
        original_shape = X.shape
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
        
        # Reshape back
        X_balanced = X_balanced.reshape(-1, original_shape[1], original_shape[2])
        
        logger.info(f"SMOTE applied - Original: {len(X)}, Balanced: {len(X_balanced)}")
        
        return X_balanced, y_balanced


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.study = None
        self.best_params = None
        
    def objective(self, trial, X_train, y_train, X_val, y_val, model_type='lstm'):
        """Objective function for Optuna"""
        
        # Suggest hyperparameters
        config_dict = {
            'sequence_length': X_train.shape[1],
            'n_features': X_train.shape[2],
            'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else 2,
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dense_dropout': trial.suggest_uniform('dense_dropout', 0.2, 0.5),
            'l1_reg': trial.suggest_loguniform('l1_reg', 1e-5, 1e-2),
            'l2_reg': trial.suggest_loguniform('l2_reg', 1e-5, 1e-2),
            'epochs': 50  # Reduced for tuning
        }
        
        if model_type == 'lstm':
            config_dict['lstm_units'] = [
                trial.suggest_int('lstm_units_1', 64, 256),
                trial.suggest_int('lstm_units_2', 32, 128),
                trial.suggest_int('lstm_units_3', 16, 64)
            ]
            config_dict['lstm_dropout'] = trial.suggest_uniform('lstm_dropout', 0.2, 0.5)
        elif model_type == 'gru':
            config_dict['gru_units'] = [
                trial.suggest_int('gru_units_1', 64, 256),
                trial.suggest_int('gru_units_2', 32, 128),
                trial.suggest_int('gru_units_3', 16, 64)
            ]
        elif model_type == 'cnn':
            config_dict['cnn_filters'] = [
                trial.suggest_int('cnn_filters_1', 32, 128),
                trial.suggest_int('cnn_filters_2', 64, 256),
                trial.suggest_int('cnn_filters_3', 128, 512)
            ]
        
        # Create model config
        model_config = ModelConfig(config_dict)
        
        # Build and train model
        if model_type == 'lstm':
            model_obj = LSTMModel(model_config)
        elif model_type == 'gru':
            model_obj = GRUModel(model_config)
        elif model_type == 'cnn':
            model_obj = CNNModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model_obj.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Add Optuna pruning callback
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, 'val_auc')
        ]
        
        history = model_obj.model.fit(
            X_train, y_train,
            batch_size=config_dict['batch_size'],
            epochs=config_dict['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Return best validation AUC
        best_auc = max(history.history['val_auc'])
        
        return best_auc
    
    def tune(self, X_train, y_train, X_val, y_val, model_type='lstm'):
        """Run hyperparameter tuning"""
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, model_type),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best validation AUC: {self.study.best_value:.4f}")
        
        return self.best_params


class CrossValidator:
    """Cross-validation for model evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.cv_results = {}
        
    def perform_cv(self, X, y, model_type='lstm', model_config=None):
        """Perform cross-validation"""
        logger.info(f"Performing {self.config.cv_folds}-fold cross-validation for {model_type}...")
        
        if self.config.cv_type == 'stratified':
            # For classification with stratification
            if len(y.shape) > 1:
                y_for_split = np.argmax(y, axis=1)
            else:
                y_for_split = y
            kfold = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = kfold.split(X, y_for_split)
        else:
            # Time series split
            kfold = TimeSeriesSplit(n_splits=self.config.cv_folds)
            splits = kfold.split(X)
        
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build model
            if model_type == 'lstm':
                model_obj = LSTMModel(model_config)
            elif model_type == 'gru':
                model_obj = GRUModel(model_config)
            elif model_type == 'cnn':
                model_obj = CNNModel(model_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train
            model_obj.build_model(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))
            model_obj.model.fit(
                X_train_fold, y_train_fold,
                batch_size=model_config.batch_size,
                epochs=model_config.epochs,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=self.config.early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0
                    )
                ],
                verbose=0
            )
            
            # Evaluate
            y_pred = model_obj.model.predict(X_val_fold, verbose=0)
            
            if len(y_val_fold.shape) > 1:
                y_val_classes = np.argmax(y_val_fold, axis=1)
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_val_classes = y_val_fold
                y_pred_classes = (y_pred > 0.5).astype(int).ravel()
            
            # Calculate metrics
            fold_scores['accuracy'].append(accuracy_score(y_val_classes, y_pred_classes))
            fold_scores['precision'].append(precision_score(y_val_classes, y_pred_classes, average='weighted'))
            fold_scores['recall'].append(recall_score(y_val_classes, y_pred_classes, average='weighted'))
            fold_scores['f1'].append(f1_score(y_val_classes, y_pred_classes, average='weighted'))
            
            try:
                fold_scores['auc'].append(roc_auc_score(y_val_fold, y_pred, multi_class='ovr', average='weighted'))
            except:
                fold_scores['auc'].append(0.0)
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in fold_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        self.cv_results[model_type] = cv_results
        
        logger.info(f"Cross-validation results for {model_type}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            logger.info(f"  {metric}: {cv_results[f'{metric}_mean']:.4f} (+/- {cv_results[f'{metric}_std']:.4f})")
        
        return cv_results


class AdvancedModelTrainer:
    """
    Comprehensive model training pipeline with hyperparameter tuning and cross-validation
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.data_preparator = DataPreparator()
        self.hyperparameter_tuner = HyperparameterTuner(self.config)
        self.cross_validator = CrossValidator(self.config)
        
        self.trained_models = {}
        self.training_history = {}
        self.evaluation_results = {}
        
        # Create save directory
        Path(self.config.model_save_dir).mkdir(exist_ok=True, parents=True)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    sequence_length: int = 24,
                    create_sequences: bool = True) -> Tuple:
        """Prepare data for training"""
        logger.info("Preparing data...")
        
        # Create sequences if needed
        if create_sequences and len(X.shape) == 2:
            X, y = self.data_preparator.create_sequences(X, y, sequence_length)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_preparator.split_data(
            X, y,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state
        )
        
        # Handle class imbalance
        if self.config.use_smote:
            X_train, y_train = self.data_preparator.apply_smote(X_train, y_train)
        
        # Calculate class weights
        class_weights = None
        if self.config.use_class_weights:
            class_weights = self.data_preparator.calculate_class_weights(y_train)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    
    def train_single_model(self, model_type: str,
                          X_train, y_train,
                          X_val, y_val,
                          model_config: Optional[ModelConfig] = None,
                          class_weights: Optional[Dict] = None):
        """Train a single model"""
        logger.info(f"Training {model_type} model...")
        
        # Use provided config or create default
        if model_config is None:
            model_config = ModelConfig({
                'sequence_length': X_train.shape[1],
                'n_features': X_train.shape[2],
                'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else 2
            })
        
        # Hyperparameter tuning
        if self.config.use_hyperparameter_tuning:
            best_params = self.hyperparameter_tuner.tune(
                X_train, y_train, X_val, y_val, model_type
            )
            
            # Update config with best params
            for key, value in best_params.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        # Build model
        if model_type == 'lstm':
            model_obj = LSTMModel(model_config)
        elif model_type == 'gru':
            model_obj = GRUModel(model_config)
        elif model_type == 'cnn':
            model_obj = CNNModel(model_config)
        elif model_type == 'transformer':
            model_obj = TransformerModel(model_config)
        elif model_type == 'hybrid':
            model_obj = HybridModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        history = model_obj.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_save_path = f"{self.config.model_save_dir}/{model_type}_model.h5"
        model_obj.model.save(model_save_path)
        
        # Store
        self.trained_models[model_type] = model_obj
        self.training_history[model_type] = history.history
        
        logger.info(f"{model_type} model training completed and saved")
        
        return model_obj, history
    
    def train_all_models(self, X_train, y_train, X_val, y_val,
                        class_weights: Optional[Dict] = None):
        """Train all configured models"""
        logger.info("Training all models...")
        
        for model_type in self.config.models_to_train:
            try:
                self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, class_weights=class_weights
                )
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
        
        logger.info("All models trained successfully")
    
    def perform_cross_validation(self, X, y):
        """Perform cross-validation for all models"""
        if not self.config.use_cv:
            logger.info("Cross-validation disabled")
            return
        
        logger.info("Performing cross-validation...")
        
        for model_type in self.config.models_to_train:
            model_config = ModelConfig({
                'sequence_length': X.shape[1],
                'n_features': X.shape[2],
                'n_classes': y.shape[1] if len(y.shape) > 1 else 2,
                'epochs': 30  # Reduced for CV
            })
            
            cv_results = self.cross_validator.perform_cv(X, y, model_type, model_config)
            self.evaluation_results[f'{model_type}_cv'] = cv_results
    
    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create and train ensemble model"""
        if not self.config.ensemble_models:
            logger.info("Ensemble training disabled")
            return None
        
        logger.info("Creating ensemble model...")
        
        model_config = ModelConfig({
            'sequence_length': X_train.shape[1],
            'n_features': X_train.shape[2],
            'n_classes': y_train.shape[1] if len(y_train.shape) > 1 else 2
        })
        
        ensemble = EnsembleModel(model_config)
        histories = ensemble.train_all(X_train, y_train, X_val, y_val)
        
        # Save ensemble
        ensemble.save_ensemble(f"{self.config.model_save_dir}/ensemble")
        
        self.trained_models['ensemble'] = ensemble
        self.training_history['ensemble'] = histories
        
        logger.info("Ensemble model created and trained")
        
        return ensemble
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History for All Models', fontsize=16)
        
        metrics = ['loss', 'accuracy', 'auc', 'precision']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for model_name, history in self.training_history.items():
                if model_name == 'ensemble':
                    continue
                
                if metric in history:
                    ax.plot(history[metric], label=f'{model_name} train')
                if f'val_{metric}' in history:
                    ax.plot(history[f'val_{metric}'], label=f'{model_name} val', linestyle='--')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.savefig(f"{self.config.model_save_dir}/training_history.png", dpi=300)
        
        plt.close()
    
    def save_training_report(self):
        """Save comprehensive training report"""
        report = {
            'config': self.config.__dict__,
            'training_history': {k: {key: [float(val) for val in values] 
                                    for key, values in v.items()} 
                                for k, v in self.training_history.items() if k != 'ensemble'},
            'evaluation_results': self.evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = f"{self.config.model_save_dir}/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")


# Main execution
if __name__ == "__main__":
    # Example usage
    config = TrainingConfig({
        'test_size': 0.2,
        'validation_size': 0.2,
        'use_hyperparameter_tuning': False,  # Set to True for tuning
        'models_to_train': ['lstm', 'gru', 'cnn'],
        'ensemble_models': True
    })
    
    trainer = AdvancedModelTrainer(config)
    
    # Load and prepare data
    # X = np.load('features.npy')
    # y = np.load('labels.npy')
    
    # X_train, X_val, X_test, y_train, y_val, y_test, class_weights = trainer.prepare_data(X, y)
    
    # Train all models
    # trainer.train_all_models(X_train, y_train, X_val, y_val, class_weights)
    
    # Create ensemble
    # ensemble = trainer.create_ensemble(X_train, y_train, X_val, y_val)
    
    # Plot and save results
    # trainer.plot_training_history()
    # trainer.save_training_report()