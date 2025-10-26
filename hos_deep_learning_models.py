"""
Code created by: Balaam Ibencho
Date:01/10/2025
Regards:Hosoptima.com
Advanced Deep Learning Models for HOS Violation Prediction
Includes LSTM, GRU, CNN, Transformer, and Ensemble architectures
I expect to add a final combinatorial hybrid model too.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_learning_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ModelConfig:
    """Configuration for deep learning models"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # Model architecture parameters
        self.sequence_length = config.get('sequence_length', 24)
        self.n_features = config.get('n_features', 50)
        self.n_classes = config.get('n_classes', 4)  # 4 types of HOS violations
        
        # LSTM parameters
        self.lstm_units = config.get('lstm_units', [128, 64, 32])
        self.lstm_dropout = config.get('lstm_dropout', 0.3)
        self.lstm_recurrent_dropout = config.get('lstm_recurrent_dropout', 0.2)
        
        # GRU parameters
        self.gru_units = config.get('gru_units', [128, 64, 32])
        self.gru_dropout = config.get('gru_dropout', 0.3)
        
        # CNN parameters
        self.cnn_filters = config.get('cnn_filters', [64, 128, 256])
        self.cnn_kernel_size = config.get('cnn_kernel_size', 3)
        self.cnn_pool_size = config.get('cnn_pool_size', 2)
        
        # Dense layer parameters
        self.dense_units = config.get('dense_units', [256, 128, 64])
        self.dense_dropout = config.get('dense_dropout', 0.4)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Regularization
        self.l1_reg = config.get('l1_reg', 0.01)
        self.l2_reg = config.get('l2_reg', 0.01)
        
        # Advanced features
        self.use_attention = config.get('use_attention', True)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.use_residual = config.get('use_residual', True)


class AttentionLayer(layers.Layer):
    """Custom attention layer for sequence models"""
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Apply attention
        weighted_input = x * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class LSTMModel:
    """Advanced LSTM model for HOS violation prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        logger.info("Building LSTM model...")
        
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1) or self.config.use_attention
            
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.lstm_dropout,
                recurrent_dropout=self.config.lstm_recurrent_dropout,
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'lstm_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Attention mechanism
        if self.config.use_attention:
            x = AttentionLayer(units=64, name='attention')(x)
        
        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.Dropout(self.config.dense_dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.config.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = BinaryCrossentropy()
        else:
            outputs = layers.Dense(self.config.n_classes, activation='softmax', name='output')(x)
            loss = CategoricalCrossentropy()
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_HOS_Model')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"LSTM model built with {model.count_params()} parameters")
        
        return model
    
    def get_callbacks(self, model_name: str = 'lstm_model') -> List[callbacks.Callback]:
        """Get training callbacks"""
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f'{model_name}_best.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=f'./logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Starting LSTM model training...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=self.get_callbacks('lstm_model'),
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        
        return self.history


class GRUModel:
    """Advanced GRU model for HOS violation prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build GRU model architecture"""
        logger.info("Building GRU model...")
        
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # GRU layers
        for i, units in enumerate(self.config.gru_units):
            return_sequences = (i < len(self.config.gru_units) - 1) or self.config.use_attention
            
            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=self.config.gru_dropout,
                recurrent_dropout=0.2,
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'gru_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_gru_{i+1}')(x)
        
        # Attention mechanism
        if self.config.use_attention:
            x = AttentionLayer(units=64, name='attention')(x)
        
        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.Dropout(self.config.dense_dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.config.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = BinaryCrossentropy()
        else:
            outputs = layers.Dense(self.config.n_classes, activation='softmax', name='output')(x)
            loss = CategoricalCrossentropy()
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='GRU_HOS_Model')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"GRU model built with {model.count_params()} parameters")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the GRU model"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Starting GRU model training...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            callbacks.ModelCheckpoint('gru_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("GRU model training completed")
        
        return self.history


class CNNModel:
    """Advanced 1D CNN model for HOS violation prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build 1D CNN model architecture"""
        logger.info("Building CNN model...")
        
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Convolutional blocks
        for i, filters in enumerate(self.config.cnn_filters):
            # Conv layer
            x = layers.Conv1D(
                filters,
                kernel_size=self.config.cnn_kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'conv_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            # Max pooling
            x = layers.MaxPooling1D(
                pool_size=self.config.cnn_pool_size,
                name=f'pool_{i+1}'
            )(x)
            
            x = layers.Dropout(0.25, name=f'dropout_conv_{i+1}')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.Dropout(self.config.dense_dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.config.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = BinaryCrossentropy()
        else:
            outputs = layers.Dense(self.config.n_classes, activation='softmax', name='output')(x)
            loss = CategoricalCrossentropy()
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_HOS_Model')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"CNN model built with {model.count_params()} parameters")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the CNN model"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Starting CNN model training...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            callbacks.ModelCheckpoint('cnn_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("CNN model training completed")
        
        return self.history


class TransformerModel:
    """Transformer-based model for HOS violation prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """Transformer encoder block"""
        # Multi-head attention
        x = layers.MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        # Feed forward network
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x + res
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build Transformer model architecture"""
        logger.info("Building Transformer model...")
        
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding
        x = inputs
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=input_shape[0],
            output_dim=input_shape[1]
        )(positions)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(3):
            x = self.transformer_encoder(
                x,
                head_size=64,
                num_heads=4,
                ff_dim=128,
                dropout=0.1
            )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config.dense_dropout)(x)
        
        # Output layer
        if self.config.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = BinaryCrossentropy()
        else:
            outputs = layers.Dense(self.config.n_classes, activation='softmax', name='output')(x)
            loss = CategoricalCrossentropy()
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer_HOS_Model')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"Transformer model built with {model.count_params()} parameters")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Transformer model"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Starting Transformer model training...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            callbacks.ModelCheckpoint('transformer_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Transformer model training completed")
        
        return self.history


class HybridModel:
    """Hybrid model combining CNN and LSTM"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build hybrid CNN-LSTM model"""
        logger.info("Building Hybrid CNN-LSTM model...")
        
        inputs = keras.Input(shape=input_shape)
        
        # CNN branch
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # LSTM branch
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Output
        if self.config.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = BinaryCrossentropy()
        else:
            outputs = layers.Dense(self.config.n_classes, activation='softmax', name='output')(x)
            loss = CategoricalCrossentropy()
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_LSTM_Model')
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"Hybrid model built with {model.count_params()} parameters")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Hybrid model"""
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Starting Hybrid model training...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            callbacks.ModelCheckpoint('hybrid_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Hybrid model training completed")
        
        return self.history


class EnsembleModel:
    """Ensemble of multiple deep learning models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.weights = None
        
    def add_model(self, name: str, model):
        """Add a model to the ensemble"""
        self.models[name] = model
        logger.info(f"Added {name} to ensemble")
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble"""
        logger.info("Training ensemble models...")
        
        histories = {}
        
        # Train LSTM
        lstm_model = LSTMModel(self.config)
        lstm_model.train(X_train, y_train, X_val, y_val)
        self.add_model('lstm', lstm_model)
        histories['lstm'] = lstm_model.history
        
        # Train GRU
        gru_model = GRUModel(self.config)
        gru_model.train(X_train, y_train, X_val, y_val)
        self.add_model('gru', gru_model)
        histories['gru'] = gru_model.history
        
        # Train CNN
        cnn_model = CNNModel(self.config)
        cnn_model.train(X_train, y_train, X_val, y_val)
        self.add_model('cnn', cnn_model)
        histories['cnn'] = cnn_model.history
        
        logger.info("All ensemble models trained")
        
        return histories
    
    def predict(self, X, method='average'):
        """Make predictions using ensemble"""
        predictions = []
        
        for name, model_obj in self.models.items():
            if hasattr(model_obj, 'model') and model_obj.model is not None:
                pred = model_obj.model.predict(X, verbose=0)
                predictions.append(pred)
                logger.info(f"Got predictions from {name}")
        
        if len(predictions) == 0:
            raise ValueError("No trained models in ensemble")
        
        predictions = np.array(predictions)
        
        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'weighted':
            if self.weights is None:
                self.weights = np.ones(len(predictions)) / len(predictions)
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif method == 'max':
            ensemble_pred = np.max(predictions, axis=0)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def save_ensemble(self, directory: str):
        """Save all models in ensemble"""
        Path(directory).mkdir(exist_ok=True)
        
        for name, model_obj in self.models.items():
            if hasattr(model_obj, 'model') and model_obj.model is not None:
                model_obj.model.save(f'{directory}/{name}_model.h5')
                logger.info(f"Saved {name} model")
        
        # Save config
        with open(f'{directory}/config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_ensemble(self, directory: str):
        """Load all models in ensemble"""
        config_path = f'{directory}/config.json'
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = ModelConfig(config_dict)
        
        for model_file in Path(directory).glob('*_model.h5'):
            model_name = model_file.stem.replace('_model', '')
            model = keras.models.load_model(
                str(model_file),
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            
            if model_name == 'lstm':
                model_obj = LSTMModel(self.config)
            elif model_name == 'gru':
                model_obj = GRUModel(self.config)
            elif model_name == 'cnn':
                model_obj = CNNModel(self.config)
            else:
                continue
            
            model_obj.model = model
            self.add_model(model_name, model_obj)
            logger.info(f"Loaded {model_name} model")


# Main execution
if __name__ == "__main__":
    # Example usage
    config = ModelConfig({
        'sequence_length': 24,
        'n_features': 50,
        'n_classes': 4,
        'epochs': 100,
        'batch_size': 32
    })
    
    # Create synthetic data for testing
    # X_train = np.random.randn(1000, 24, 50)
    # y_train = np.random.randint(0, 4, (1000, 4))
    # X_val = np.random.randn(200, 24, 50)
    # y_val = np.random.randint(0, 4, (200, 4))
    
    # Build and train ensemble
    # ensemble = EnsembleModel(config)
    # histories = ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # Make predictions
    # predictions = ensemble.predict(X_val, method='average')
    
    # Save ensemble
    # ensemble.save_ensemble('./models/ensemble')
