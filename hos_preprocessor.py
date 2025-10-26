"""
Advanced Data Preprocessor for HOS Violation Prediction System
Handles missing values, normalization, feature scaling, outlier detection
Production-ready with comprehensive data quality checks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from datetime import datetime
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PreprocessingConfig:
    """Configuration for preprocessing operations"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # Missing value handling
        self.missing_strategy_numeric = config.get('missing_strategy_numeric', 'median')
        self.missing_strategy_categorical = config.get('missing_strategy_categorical', 'most_frequent')
        self.missing_threshold = config.get('missing_threshold', 0.5)
        
        # Scaling
        self.scaling_method = config.get('scaling_method', 'standard')
        
        # Outlier detection
        self.outlier_method = config.get('outlier_method', 'iqr')
        self.outlier_threshold = config.get('outlier_threshold', 1.5)
        
        # Encoding
        self.encoding_method = config.get('encoding_method', 'onehot')
        self.max_categories = config.get('max_categories', 20)
        
        # Feature selection
        self.variance_threshold = config.get('variance_threshold', 0.01)
        
        # Date handling
        self.extract_date_features = config.get('extract_date_features', True)


class MissingValueHandler:
    """Handle missing values with multiple strategies"""
    
    def __init__(self, strategy: str = 'median'):
        self.strategy = strategy
        self.imputers = {}
        
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing value patterns"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2),
                    'dtype': str(df[col].dtype)
                }
        
        logger.info(f"Found missing values in {len(missing_info)} columns")
        return missing_info
    
    def handle_numeric_missing(self, df: pd.DataFrame, 
                              columns: List[str],
                              strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in numeric columns"""
        df_copy = df.copy()
        
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            logger.warning(f"Unknown strategy {strategy}, using median")
            imputer = SimpleImputer(strategy='median')
        
        for col in columns:
            if col in df_copy.columns and df_copy[col].isnull().any():
                try:
                    df_copy[col] = imputer.fit_transform(
                        df_copy[[col]]
                    ).ravel()
                    logger.info(f"Imputed missing values in {col}")
                except Exception as e:
                    logger.error(f"Error imputing {col}: {str(e)}")
        
        return df_copy
    
    def handle_categorical_missing(self, df: pd.DataFrame,
                                   columns: List[str],
                                   strategy: str = 'most_frequent') -> pd.DataFrame:
        """Handle missing values in categorical columns"""
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns and df_copy[col].isnull().any():
                if strategy == 'most_frequent':
                    mode_value = df_copy[col].mode()
                    if len(mode_value) > 0:
                        df_copy[col].fillna(mode_value[0], inplace=True)
                elif strategy == 'unknown':
                    df_copy[col].fillna('Unknown', inplace=True)
                elif strategy == 'forward_fill':
                    df_copy[col].fillna(method='ffill', inplace=True)
                
                logger.info(f"Handled missing values in {col}")
        
        return df_copy
    
    def remove_high_missing_columns(self, df: pd.DataFrame, 
                                   threshold: float = 0.5) -> pd.DataFrame:
        """Remove columns with high percentage of missing values"""
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        return df


class OutlierDetector:
    """Detect and handle outliers using multiple methods"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        self.method = method
        self.threshold = threshold
        self.outlier_bounds = {}
    
    def detect_outliers_iqr(self, df: pd.DataFrame, 
                           columns: List[str]) -> Dict[str, np.ndarray]:
        """Detect outliers using IQR method"""
        outliers = {}
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = outlier_mask
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {col}")
        
        return outliers
    
    def detect_outliers_zscore(self, df: pd.DataFrame,
                              columns: List[str],
                              threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """Detect outliers using Z-score method"""
        outliers = {}
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = z_scores > threshold
            
            # Expand to full length
            full_mask = pd.Series([False] * len(df), index=df.index)
            full_mask.loc[df[col].notna()] = outlier_mask
            
            outliers[col] = full_mask.values
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {col} (Z-score)")
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame,
                       outliers: Dict[str, np.ndarray],
                       method: str = 'clip') -> pd.DataFrame:
        """Handle detected outliers"""
        df_copy = df.copy()
        
        for col, outlier_mask in outliers.items():
            if method == 'clip':
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
                    logger.info(f"Clipped outliers in {col}")
            elif method == 'remove':
                df_copy = df_copy[~outlier_mask]
                logger.info(f"Removed outliers from {col}")
            elif method == 'cap':
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    df_copy.loc[df_copy[col] < lower, col] = lower
                    df_copy.loc[df_copy[col] > upper, col] = upper
        
        return df_copy


class DataScaler:
    """Scale numerical features using various methods"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
        
    def fit_transform(self, df: pd.DataFrame, 
                     columns: List[str]) -> pd.DataFrame:
        """Fit scaler and transform data"""
        df_copy = df.copy()
        
        if self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method {self.method}, using standard")
            scaler = StandardScaler()
        
        for col in columns:
            if col in df_copy.columns and df_copy[col].dtype in ['int64', 'float64']:
                try:
                    df_copy[col] = scaler.fit_transform(df_copy[[col]])
                    self.scalers[col] = scaler
                    logger.info(f"Scaled column {col} using {self.method}")
                except Exception as e:
                    logger.error(f"Error scaling {col}: {str(e)}")
        
        return df_copy
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        df_copy = df.copy()
        
        for col in columns:
            if col in self.scalers and col in df_copy.columns:
                try:
                    df_copy[col] = self.scalers[col].transform(df_copy[[col]])
                except Exception as e:
                    logger.error(f"Error transforming {col}: {str(e)}")
        
        return df_copy
    
    def inverse_transform(self, df: pd.DataFrame, 
                         columns: List[str]) -> pd.DataFrame:
        """Inverse transform scaled data"""
        df_copy = df.copy()
        
        for col in columns:
            if col in self.scalers and col in df_copy.columns:
                try:
                    df_copy[col] = self.scalers[col].inverse_transform(df_copy[[col]])
                except Exception as e:
                    logger.error(f"Error inverse transforming {col}: {str(e)}")
        
        return df_copy


class CategoricalEncoder:
    """Encode categorical variables"""
    
    def __init__(self, method: str = 'onehot', max_categories: int = 20):
        self.method = method
        self.max_categories = max_categories
        self.encoders = {}
        self.encoded_columns = {}
        
    def fit_transform(self, df: pd.DataFrame, 
                     columns: List[str]) -> pd.DataFrame:
        """Fit encoder and transform categorical columns"""
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
            
            unique_count = df_copy[col].nunique()
            
            if unique_count > self.max_categories:
                logger.warning(f"{col} has {unique_count} categories, using label encoding")
                method = 'label'
            else:
                method = self.method
            
            if method == 'label':
                encoder = LabelEncoder()
                df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))
                self.encoders[col] = encoder
                logger.info(f"Label encoded {col}")
                
            elif method == 'onehot':
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
                self.encoded_columns[col] = dummies.columns.tolist()
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy.drop(columns=[col], inplace=True)
                logger.info(f"One-hot encoded {col} into {len(dummies.columns)} columns")
                
            elif method == 'target':
                # Placeholder for target encoding (needs target variable)
                logger.warning(f"Target encoding for {col} requires target variable")
        
        return df_copy
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders"""
        df_copy = df.copy()
        
        for col in columns:
            if col in self.encoders and col in df_copy.columns:
                try:
                    df_copy[col] = self.encoders[col].transform(df_copy[col].astype(str))
                except Exception as e:
                    logger.error(f"Error transforming {col}: {str(e)}")
            
            elif col in self.encoded_columns:
                # Handle one-hot encoding
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
                
                # Ensure all expected columns are present
                for encoded_col in self.encoded_columns[col]:
                    if encoded_col not in dummies.columns:
                        dummies[encoded_col] = 0
                
                df_copy = pd.concat([df_copy, dummies[self.encoded_columns[col]]], axis=1)
                df_copy.drop(columns=[col], inplace=True)
        
        return df_copy


class DateFeatureExtractor:
    """Extract features from date/datetime columns"""
    
    @staticmethod
    def extract_date_features(df: pd.DataFrame, 
                            date_columns: List[str]) -> pd.DataFrame:
        """Extract various features from date columns"""
        df_copy = df.copy()
        
        for col in date_columns:
            if col not in df_copy.columns:
                continue
            
            # Ensure column is datetime
            if df_copy[col].dtype != 'datetime64[ns]':
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            
            # Extract features
            df_copy[f'{col}_year'] = df_copy[col].dt.year
            df_copy[f'{col}_month'] = df_copy[col].dt.month
            df_copy[f'{col}_day'] = df_copy[col].dt.day
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy[f'{col}_hour'] = df_copy[col].dt.hour
            df_copy[f'{col}_minute'] = df_copy[col].dt.minute
            df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
            df_copy[f'{col}_is_weekend'] = (df_copy[col].dt.dayofweek >= 5).astype(int)
            df_copy[f'{col}_is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
            df_copy[f'{col}_is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
            
            logger.info(f"Extracted date features from {col}")
        
        return df_copy
    
    @staticmethod
    def create_time_based_features(df: pd.DataFrame, 
                                  timestamp_col: str) -> pd.DataFrame:
        """Create time-based aggregation features"""
        df_copy = df.copy()
        
        if timestamp_col not in df_copy.columns:
            return df_copy
        
        # Ensure datetime type
        if df_copy[timestamp_col].dtype != 'datetime64[ns]':
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
        
        # Time since epoch
        df_copy[f'{timestamp_col}_epoch'] = df_copy[timestamp_col].astype(np.int64) // 10**9
        
        # Time of day categories
        hour = df_copy[timestamp_col].dt.hour
        df_copy[f'{timestamp_col}_time_of_day'] = pd.cut(
            hour,
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        return df_copy


class AdvancedDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for HOS violation prediction
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector(
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
        self.scaler = DataScaler(method=self.config.scaling_method)
        self.encoder = CategoricalEncoder(
            method=self.config.encoding_method,
            max_categories=self.config.max_categories
        )
        self.date_extractor = DateFeatureExtractor()
        
        self.preprocessing_stats = {}
        self.is_fitted = False
        
    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically identify column types"""
        column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'binary': [],
            'id': []
        }
        
        for col in df.columns:
            # Check for ID columns
            if 'id' in col.lower() or col.lower().endswith('_id'):
                column_types['id'].append(col)
                continue
            
            # Check for datetime
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                column_types['datetime'].append(col)
                continue
            
            # Check for numeric
            if df[col].dtype in ['int64', 'float64']:
                # Check if binary
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    column_types['binary'].append(col)
                else:
                    column_types['numeric'].append(col)
                continue
            
            # Check for categorical
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                avg_length = df[col].astype(str).str.len().mean()
                
                # If text is long, classify as text
                if avg_length > 50:
                    column_types['text'].append(col)
                else:
                    column_types['categorical'].append(col)
        
        logger.info(f"Identified column types: {json.dumps({k: len(v) for k, v in column_types.items()})}")
        return column_types
    
    def handle_missing_values(self, df: pd.DataFrame,
                            column_types: Dict[str, List[str]]) -> pd.DataFrame:
        """Handle missing values across all column types"""
        logger.info("Handling missing values...")
        
        # Analyze missing values
        missing_info = self.missing_handler.analyze_missing_values(df)
        self.preprocessing_stats['missing_values'] = missing_info
        
        # Remove high-missing columns
        df = self.missing_handler.remove_high_missing_columns(
            df, self.config.missing_threshold
        )
        
        # Handle numeric columns
        if column_types['numeric']:
            df = self.missing_handler.handle_numeric_missing(
                df, column_types['numeric'], 
                self.config.missing_strategy_numeric
            )
        
        # Handle categorical columns
        if column_types['categorical']:
            df = self.missing_handler.handle_categorical_missing(
                df, column_types['categorical'],
                self.config.missing_strategy_categorical
            )
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame,
                       numeric_columns: List[str]) -> pd.DataFrame:
        """Detect and handle outliers in numeric columns"""
        logger.info("Detecting and handling outliers...")
        
        if self.config.outlier_method == 'iqr':
            outliers = self.outlier_detector.detect_outliers_iqr(df, numeric_columns)
        elif self.config.outlier_method == 'zscore':
            outliers = self.outlier_detector.detect_outliers_zscore(df, numeric_columns)
        else:
            logger.warning(f"Unknown outlier method: {self.config.outlier_method}")
            return df
        
        self.preprocessing_stats['outliers'] = {
            col: int(mask.sum()) for col, mask in outliers.items()
        }
        
        # Handle outliers by clipping
        df = self.outlier_detector.handle_outliers(df, outliers, method='clip')
        
        return df
    
    def scale_features(self, df: pd.DataFrame,
                      numeric_columns: List[str]) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling numeric features...")
        
        # Exclude ID columns and binary columns from scaling
        columns_to_scale = [col for col in numeric_columns 
                           if not col.lower().endswith('_id') 
                           and df[col].nunique() > 2]
        
        if not self.is_fitted:
            df = self.scaler.fit_transform(df, columns_to_scale)
        else:
            df = self.scaler.transform(df, columns_to_scale)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame,
                          categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        if not self.is_fitted:
            df = self.encoder.fit_transform(df, categorical_columns)
        else:
            df = self.encoder.transform(df, categorical_columns)
        
        return df
    
    def extract_datetime_features(self, df: pd.DataFrame,
                                 datetime_columns: List[str]) -> pd.DataFrame:
        """Extract features from datetime columns"""
        if not datetime_columns or not self.config.extract_date_features:
            return df
        
        logger.info("Extracting datetime features...")
        df = self.date_extractor.extract_date_features(df, datetime_columns)
        
        # Create time-based features for the first datetime column
        if datetime_columns:
            df = self.date_extractor.create_time_based_features(
                df, datetime_columns[0]
            )
        
        return df
    
    def remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        logger.info("Checking for low variance features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_cols = []
        
        for col in numeric_cols:
            variance = df[col].var()
            if variance < self.config.variance_threshold:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            logger.info(f"Removing {len(low_variance_cols)} low variance features")
            df = df.drop(columns=low_variance_cols)
            self.preprocessing_stats['low_variance_removed'] = low_variance_cols
        
        return df
    
    def handle_imbalanced_data(self, df: pd.DataFrame, 
                              target_col: str,
                              method: str = 'smote') -> pd.DataFrame:
        """Handle imbalanced target variable (optional)"""
        if target_col not in df.columns:
            return df
        
        class_counts = df[target_col].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        self.preprocessing_stats['class_distribution'] = class_counts.to_dict()
        
        if imbalance_ratio > 5:
            logger.warning(f"Significant class imbalance detected. Consider using SMOTE or class weights.")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified pairs"""
        logger.info("Creating interaction features...")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Additive interaction
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                
                logger.info(f"Created interaction between {feat1} and {feat2}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame,
                     target_col: Optional[str] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline - fit and transform"""
        logger.info("Starting preprocessing pipeline (fit_transform)...")
        
        # Store original shape
        original_shape = df.shape
        logger.info(f"Original data shape: {original_shape}")
        
        # Identify column types
        column_types = self.identify_column_types(df)
        self.preprocessing_stats['column_types'] = column_types
        
        # Handle missing values
        df = self.handle_missing_values(df, column_types)
        
        # Extract datetime features before removing datetime columns
        if column_types['datetime']:
            df = self.extract_datetime_features(df, column_types['datetime'])
            # Drop original datetime columns
            df = df.drop(columns=column_types['datetime'])
        
        # Handle outliers in numeric columns
        if column_types['numeric']:
            df = self.handle_outliers(df, column_types['numeric'])
        
        # Encode categorical features
        if column_types['categorical']:
            df = self.encode_categorical(df, column_types['categorical'])
        
        # Scale numeric features
        if column_types['numeric']:
            df = self.scale_features(df, column_types['numeric'])
        
        # Remove low variance features
        df = self.remove_low_variance_features(df)
        
        # Handle imbalanced data if target provided
        if target_col:
            df = self.handle_imbalanced_data(df, target_col)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Store final statistics
        final_shape = df.shape
        logger.info(f"Final data shape: {final_shape}")
        self.preprocessing_stats['shape_change'] = {
            'original': original_shape,
            'final': final_shape
        }
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Identify column types
        column_types = self.identify_column_types(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, column_types)
        
        # Extract datetime features
        if column_types['datetime']:
            df = self.extract_datetime_features(df, column_types['datetime'])
            df = df.drop(columns=column_types['datetime'])
        
        # Encode categorical
        if column_types['categorical']:
            df = self.encode_categorical(df, column_types['categorical'])
        
        # Scale numeric
        if column_types['numeric']:
            df = self.scale_features(df, column_types['numeric'])
        
        return df
    
    def save_preprocessor(self, filepath: str):
        """Save fitted preprocessor to file"""
        preprocessor_state = {
            'config': self.config.__dict__,
            'scaler': self.scaler.scalers,
            'encoder': {
                'encoders': self.encoder.encoders,
                'encoded_columns': self.encoder.encoded_columns
            },
            'outlier_bounds': self.outlier_detector.outlier_bounds,
            'preprocessing_stats': self.preprocessing_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load fitted preprocessor from file"""
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.config = PreprocessingConfig(preprocessor_state['config'])
        self.scaler.scalers = preprocessor_state['scaler']
        self.encoder.encoders = preprocessor_state['encoder']['encoders']
        self.encoder.encoded_columns = preprocessor_state['encoder']['encoded_columns']
        self.outlier_detector.outlier_bounds = preprocessor_state['outlier_bounds']
        self.preprocessing_stats = preprocessor_state['preprocessing_stats']
        self.is_fitted = preprocessor_state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def get_preprocessing_report(self) -> str:
        """Generate comprehensive preprocessing report"""
        report = []
        report.append("=" * 80)
        report.append("PREPROCESSING REPORT")
        report.append("=" * 80)
        
        if 'shape_change' in self.preprocessing_stats:
            report.append("\nShape Changes:")
            report.append(f"  Original: {self.preprocessing_stats['shape_change']['original']}")
            report.append(f"  Final: {self.preprocessing_stats['shape_change']['final']}")
        
        if 'missing_values' in self.preprocessing_stats:
            report.append("\nMissing Values Handled:")
            for col, info in list(self.preprocessing_stats['missing_values'].items())[:10]:
                report.append(f"  {col}: {info['count']} ({info['percentage']}%)")
        
        if 'outliers' in self.preprocessing_stats:
            report.append("\nOutliers Detected:")
            for col, count in list(self.preprocessing_stats['outliers'].items())[:10]:
                report.append(f"  {col}: {count} outliers")
        
        if 'column_types' in self.preprocessing_stats:
            report.append("\nColumn Types:")
            for ctype, cols in self.preprocessing_stats['column_types'].items():
                report.append(f"  {ctype}: {len(cols)} columns")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Main execution
if __name__ == "__main__":
    # Example usage
    config = PreprocessingConfig({
        'scaling_method': 'standard',
        'encoding_method': 'onehot',
        'outlier_method': 'iqr'
    })
    
    preprocessor = AdvancedDataPreprocessor(config)
    
    # Load sample data
    # df = pd.read_csv('merged_hos_data.csv')
    
    # Preprocess
    # processed_df = preprocessor.fit_transform(df, target_col='violation_type')
    
    # Get report
    # print(preprocessor.get_preprocessing_report())
    
    # Save preprocessor
    # preprocessor.save_preprocessor('preprocessor.pkl')