"""
Advanced Feature Engineering for HOS Violation Prediction System
Creates domain-specific features, temporal patterns, aggregations
Production-ready with comprehensive feature generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HOSFeatureConfig:
    """Configuration for HOS-specific feature engineering"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # HOS regulations parameters
        self.max_daily_drive_hours = config.get('max_daily_drive_hours', 11)
        self.max_daily_duty_hours = config.get('max_daily_duty_hours', 14)
        self.max_weekly_hours = config.get('max_weekly_hours', 60)
        self.required_break_hours = config.get('required_break_hours', 0.5)
        self.max_continuous_drive_hours = config.get('max_continuous_drive_hours', 8)
        
        # Feature engineering parameters
        self.create_rolling_features = config.get('create_rolling_features', True)
        self.rolling_windows = config.get('rolling_windows', [7, 14, 30])
        self.create_lag_features = config.get('create_lag_features', True)
        self.lag_periods = config.get('lag_periods', [1, 3, 7])
        self.create_statistical_features = config.get('create_statistical_features', True)


class TemporalFeatureExtractor:
    """Extract temporal patterns and time-based features"""
    
    @staticmethod
    def create_time_series_features(df: pd.DataFrame,
                                   timestamp_col: str,
                                   value_cols: List[str],
                                   group_col: Optional[str] = None) -> pd.DataFrame:
        """Create time series features from sequential data"""
        df_copy = df.copy()
        
        if timestamp_col not in df_copy.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df_copy
        
        # Ensure sorted by timestamp
        df_copy = df_copy.sort_values(timestamp_col)
        
        for col in value_cols:
            if col not in df_copy.columns:
                continue
            
            if group_col:
                # Group-wise features
                df_copy[f'{col}_lag_1'] = df_copy.groupby(group_col)[col].shift(1)
                df_copy[f'{col}_lag_7'] = df_copy.groupby(group_col)[col].shift(7)
                df_copy[f'{col}_diff_1'] = df_copy.groupby(group_col)[col].diff(1)
                df_copy[f'{col}_pct_change'] = df_copy.groupby(group_col)[col].pct_change()
                
                # Rolling statistics
                df_copy[f'{col}_rolling_mean_7'] = df_copy.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
                df_copy[f'{col}_rolling_std_7'] = df_copy.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).std()
                )
                df_copy[f'{col}_rolling_max_7'] = df_copy.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).max()
                )
                df_copy[f'{col}_rolling_min_7'] = df_copy.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).min()
                )
            else:
                # Global features
                df_copy[f'{col}_lag_1'] = df_copy[col].shift(1)
                df_copy[f'{col}_diff_1'] = df_copy[col].diff(1)
                df_copy[f'{col}_rolling_mean_7'] = df_copy[col].rolling(window=7, min_periods=1).mean()
            
            logger.info(f"Created time series features for {col}")
        
        return df_copy
    
    @staticmethod
    def create_cyclic_features(df: pd.DataFrame,
                             timestamp_col: str) -> pd.DataFrame:
        """Create cyclic encoding for temporal features"""
        df_copy = df.copy()
        
        if timestamp_col not in df_copy.columns:
            return df_copy
        
        # Ensure datetime
        if df_copy[timestamp_col].dtype != 'datetime64[ns]':
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
        
        # Cyclic encoding for hour
        hour = df_copy[timestamp_col].dt.hour
        df_copy[f'{timestamp_col}_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df_copy[f'{timestamp_col}_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Cyclic encoding for day of week
        dayofweek = df_copy[timestamp_col].dt.dayofweek
        df_copy[f'{timestamp_col}_dow_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        df_copy[f'{timestamp_col}_dow_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        
        # Cyclic encoding for month
        month = df_copy[timestamp_col].dt.month
        df_copy[f'{timestamp_col}_month_sin'] = np.sin(2 * np.pi * month / 12)
        df_copy[f'{timestamp_col}_month_cos'] = np.cos(2 * np.pi * month / 12)
        
        logger.info("Created cyclic temporal features")
        
        return df_copy
    
    @staticmethod
    def create_holiday_features(df: pd.DataFrame,
                               timestamp_col: str,
                               country: str = 'US') -> pd.DataFrame:
        """Create features for holidays and special days"""
        df_copy = df.copy()
        
        if timestamp_col not in df_copy.columns:
            return df_copy
        
        # Ensure datetime
        if df_copy[timestamp_col].dtype != 'datetime64[ns]':
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
        
        # Is weekend
        df_copy[f'{timestamp_col}_is_weekend'] = (df_copy[timestamp_col].dt.dayofweek >= 5).astype(int)
        
        # Is month end/start
        df_copy[f'{timestamp_col}_is_month_end'] = df_copy[timestamp_col].dt.is_month_end.astype(int)
        df_copy[f'{timestamp_col}_is_month_start'] = df_copy[timestamp_col].dt.is_month_start.astype(int)
        
        # Days until weekend
        df_copy[f'{timestamp_col}_days_to_weekend'] = 4 - df_copy[timestamp_col].dt.dayofweek
        df_copy.loc[df_copy[f'{timestamp_col}_days_to_weekend'] < 0, f'{timestamp_col}_days_to_weekend'] = 0
        
        return df_copy


class HOSComplianceFeatures:
    """Generate HOS regulation compliance-specific features"""
    
    def __init__(self, config: HOSFeatureConfig):
        self.config = config
    
    def calculate_hours_remaining(self, df: pd.DataFrame,
                                  hours_worked_col: str = 'hours_worked') -> pd.DataFrame:
        """Calculate remaining hours under HOS regulations"""
        df_copy = df.copy()
        
        if hours_worked_col not in df_copy.columns:
            logger.warning(f"Column {hours_worked_col} not found")
            return df_copy
        
        # Daily driving hours remaining
        df_copy['daily_drive_hours_remaining'] = self.config.max_daily_drive_hours - df_copy[hours_worked_col]
        df_copy['daily_drive_hours_remaining'] = df_copy['daily_drive_hours_remaining'].clip(lower=0)
        
        # Percentage of daily limit used
        df_copy['daily_drive_hours_pct_used'] = (df_copy[hours_worked_col] / self.config.max_daily_drive_hours * 100).clip(upper=100)
        
        # Risk score based on hours worked
        df_copy['hours_worked_risk_score'] = np.where(
            df_copy[hours_worked_col] >= self.config.max_daily_drive_hours * 0.9,
            3,  # High risk
            np.where(
                df_copy[hours_worked_col] >= self.config.max_daily_drive_hours * 0.75,
                2,  # Medium risk
                1   # Low risk
            )
        )
        
        logger.info("Created HOS hours remaining features")
        
        return df_copy
    
    def calculate_break_compliance(self, df: pd.DataFrame,
                                   breaks_col: str = 'breaks_taken',
                                   hours_col: str = 'hours_worked') -> pd.DataFrame:
        """Calculate break compliance features"""
        df_copy = df.copy()
        
        if breaks_col not in df_copy.columns or hours_col not in df_copy.columns:
            logger.warning(f"Required columns not found for break compliance")
            return df_copy
        
        # Break ratio (breaks per hour worked)
        df_copy['break_ratio'] = df_copy[breaks_col] / (df_copy[hours_col] + 1e-6)
        
        # Is break compliant
        required_breaks = (df_copy[hours_col] / self.config.max_continuous_drive_hours).apply(np.floor)
        df_copy['break_deficit'] = (required_breaks - df_copy[breaks_col]).clip(lower=0)
        df_copy['is_break_compliant'] = (df_copy['break_deficit'] == 0).astype(int)
        
        # Time since last break (simulated - would need actual timestamp data)
        df_copy['hours_since_last_break'] = df_copy.groupby('driver_id')[hours_col].cumsum() % self.config.max_continuous_drive_hours if 'driver_id' in df_copy.columns else 0
        
        logger.info("Created break compliance features")
        
        return df_copy
    
    def calculate_weekly_compliance(self, df: pd.DataFrame,
                                    driver_col: str = 'driver_id',
                                    hours_col: str = 'hours_worked',
                                    date_col: str = 'timestamp') -> pd.DataFrame:
        """Calculate weekly hours compliance features"""
        df_copy = df.copy()
        
        if driver_col not in df_copy.columns or hours_col not in df_copy.columns:
            logger.warning("Required columns for weekly compliance not found")
            return df_copy
        
        # Ensure date column is datetime
        if date_col in df_copy.columns:
            if df_copy[date_col].dtype != 'datetime64[ns]':
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            
            # Calculate week number
            df_copy['week_number'] = df_copy[date_col].dt.isocalendar().week
            
            # Calculate cumulative weekly hours
            df_copy['weekly_hours_cumsum'] = df_copy.groupby([driver_col, 'week_number'])[hours_col].cumsum()
            
            # Weekly hours remaining
            df_copy['weekly_hours_remaining'] = (self.config.max_weekly_hours - df_copy['weekly_hours_cumsum']).clip(lower=0)
            
            # Weekly compliance status
            df_copy['weekly_hours_compliant'] = (df_copy['weekly_hours_cumsum'] <= self.config.max_weekly_hours).astype(int)
            
            logger.info("Created weekly compliance features")
        
        return df_copy
    
    def create_violation_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk features for violations"""
        df_copy = df.copy()
        
        # Composite risk score
        risk_components = []
        
        if 'hours_worked_risk_score' in df_copy.columns:
            risk_components.append(df_copy['hours_worked_risk_score'])
        
        if 'is_break_compliant' in df_copy.columns:
            risk_components.append((1 - df_copy['is_break_compliant']) * 2)
        
        if 'weekly_hours_compliant' in df_copy.columns:
            risk_components.append((1 - df_copy['weekly_hours_compliant']) * 3)
        
        if risk_components:
            df_copy['overall_violation_risk'] = np.sum(risk_components, axis=0)
            df_copy['overall_violation_risk'] = df_copy['overall_violation_risk'].clip(upper=10)
        
        logger.info("Created violation risk features")
        
        return df_copy


class AggregationFeatures:
    """Create aggregation features across different dimensions"""
    
    @staticmethod
    def create_driver_aggregations(df: pd.DataFrame,
                                   driver_col: str = 'driver_id',
                                   agg_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create driver-level aggregation features"""
        df_copy = df.copy()
        
        if driver_col not in df_copy.columns:
            logger.warning(f"Driver column {driver_col} not found")
            return df_copy
        
        if agg_cols is None:
            agg_cols = ['hours_worked', 'breaks_taken', 'miles_driven']
        
        # Filter to existing columns
        agg_cols = [col for col in agg_cols if col in df_copy.columns]
        
        for col in agg_cols:
            # Mean
            df_copy[f'driver_{col}_mean'] = df_copy.groupby(driver_col)[col].transform('mean')
            
            # Std
            df_copy[f'driver_{col}_std'] = df_copy.groupby(driver_col)[col].transform('std')
            
            # Max
            df_copy[f'driver_{col}_max'] = df_copy.groupby(driver_col)[col].transform('max')
            
            # Min
            df_copy[f'driver_{col}_min'] = df_copy.groupby(driver_col)[col].transform('min')
            
            # Count
            df_copy[f'driver_{col}_count'] = df_copy.groupby(driver_col)[col].transform('count')
            
            # Deviation from mean
            df_copy[f'driver_{col}_deviation'] = df_copy[col] - df_copy[f'driver_{col}_mean']
            
            logger.info(f"Created driver aggregations for {col}")
        
        return df_copy
    
    @staticmethod
    def create_vehicle_aggregations(df: pd.DataFrame,
                                    vehicle_col: str = 'vehicle_id',
                                    agg_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create vehicle-level aggregation features"""
        df_copy = df.copy()
        
        if vehicle_col not in df_copy.columns:
            logger.warning(f"Vehicle column {vehicle_col} not found")
            return df_copy
        
        if agg_cols is None:
            agg_cols = ['hours_worked', 'miles_driven']
        
        agg_cols = [col for col in agg_cols if col in df_copy.columns]
        
        for col in agg_cols:
            df_copy[f'vehicle_{col}_mean'] = df_copy.groupby(vehicle_col)[col].transform('mean')
            df_copy[f'vehicle_{col}_std'] = df_copy.groupby(vehicle_col)[col].transform('std')
            df_copy[f'vehicle_{col}_max'] = df_copy.groupby(vehicle_col)[col].transform('max')
            
            logger.info(f"Created vehicle aggregations for {col}")
        
        return df_copy
    
    @staticmethod
    def create_carrier_aggregations(df: pd.DataFrame,
                                    carrier_col: str = 'carrier_id',
                                    agg_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create carrier-level aggregation features"""
        df_copy = df.copy()
        
        if carrier_col not in df_copy.columns:
            logger.warning(f"Carrier column {carrier_col} not found")
            return df_copy
        
        if agg_cols is None:
            agg_cols = ['hours_worked', 'violations']
        
        agg_cols = [col for col in agg_cols if col in df_copy.columns]
        
        for col in agg_cols:
            df_copy[f'carrier_{col}_mean'] = df_copy.groupby(carrier_col)[col].transform('mean')
            df_copy[f'carrier_{col}_median'] = df_copy.groupby(carrier_col)[col].transform('median')
            
            logger.info(f"Created carrier aggregations for {col}")
        
        return df_copy
    
    @staticmethod
    def create_time_window_aggregations(df: pd.DataFrame,
                                       timestamp_col: str,
                                       value_cols: List[str],
                                       windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window aggregations"""
        df_copy = df.copy()
        
        if timestamp_col not in df_copy.columns:
            return df_copy
        
        # Ensure sorted
        df_copy = df_copy.sort_values(timestamp_col)
        
        for window in windows:
            for col in value_cols:
                if col not in df_copy.columns:
                    continue
                
                # Rolling mean
                df_copy[f'{col}_rolling_mean_{window}d'] = df_copy[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling sum
                df_copy[f'{col}_rolling_sum_{window}d'] = df_copy[col].rolling(
                    window=window, min_periods=1
                ).sum()
                
                # Rolling max
                df_copy[f'{col}_rolling_max_{window}d'] = df_copy[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                logger.info(f"Created {window}-day rolling features for {col}")
        
        return df_copy


class StatisticalFeatures:
    """Create statistical features from data"""
    
    @staticmethod
    def create_distribution_features(df: pd.DataFrame,
                                    numeric_cols: List[str]) -> pd.DataFrame:
        """Create features based on statistical distributions"""
        df_copy = df.copy()
        
        for col in numeric_cols:
            if col not in df_copy.columns:
                continue
            
            # Skewness
            df_copy[f'{col}_skew'] = df_copy[col].skew() if len(df_copy[col].dropna()) > 0 else 0
            
            # Kurtosis
            df_copy[f'{col}_kurtosis'] = df_copy[col].kurtosis() if len(df_copy[col].dropna()) > 0 else 0
            
            # Percentile features
            df_copy[f'{col}_percentile_25'] = df_copy[col].quantile(0.25)
            df_copy[f'{col}_percentile_75'] = df_copy[col].quantile(0.75)
            
            logger.info(f"Created distribution features for {col}")
        
        return df_copy
    
    @staticmethod
    def create_ratio_features(df: pd.DataFrame,
                             ratio_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create ratio features between column pairs"""
        df_copy = df.copy()
        
        for col1, col2 in ratio_pairs:
            if col1 in df_copy.columns and col2 in df_copy.columns:
                # Avoid division by zero
                df_copy[f'{col1}_to_{col2}_ratio'] = df_copy[col1] / (df_copy[col2] + 1e-6)
                
                logger.info(f"Created ratio feature: {col1}/{col2}")
        
        return df_copy
    
    @staticmethod
    def create_binned_features(df: pd.DataFrame,
                              cols_to_bin: List[str],
                              n_bins: int = 5) -> pd.DataFrame:
        """Create binned versions of continuous features"""
        df_copy = df.copy()
        
        for col in cols_to_bin:
            if col not in df_copy.columns:
                continue
            
            # Create quantile-based bins
            df_copy[f'{col}_binned'] = pd.qcut(
                df_copy[col], 
                q=n_bins, 
                labels=False, 
                duplicates='drop'
            )
            
            logger.info(f"Created binned feature for {col}")
        
        return df_copy


class DimensionalityReducer:
    """Reduce feature dimensionality using PCA and other methods"""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = None
        self.feature_columns = None
    
    def fit_transform(self, df: pd.DataFrame,
                     feature_cols: List[str]) -> pd.DataFrame:
        """Apply PCA to reduce dimensions"""
        df_copy = df.copy()
        
        # Filter to numeric columns
        feature_cols = [col for col in feature_cols 
                       if col in df_copy.columns and df_copy[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            logger.warning("No numeric features found for PCA")
            return df_copy
        
        self.feature_columns = feature_cols
        
        # Extract features
        X = df_copy[feature_cols].fillna(0)
        
        # Fit PCA
        n_components = min(self.n_components, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_components)
        
        pca_features = self.pca.fit_transform(X)
        
        # Add PCA features to dataframe
        for i in range(n_components):
            df_copy[f'pca_component_{i+1}'] = pca_features[:, i]
        
        logger.info(f"Created {n_components} PCA components explaining {self.pca.explained_variance_ratio_.sum():.2%} variance")
        
        return df_copy
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted PCA"""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_transform first.")
        
        df_copy = df.copy()
        X = df_copy[self.feature_columns].fillna(0)
        
        pca_features = self.pca.transform(X)
        
        for i in range(self.pca.n_components_):
            df_copy[f'pca_component_{i+1}'] = pca_features[:, i]
        
        return df_copy


class FeatureSelector:
    """Select most important features"""
    
    def __init__(self, method: str = 'mutual_info', k: int = 50):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Fit feature selector and return selected features"""
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].fillna(0)
        
        k = min(self.k, len(numeric_cols))
        
        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif self.method == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=k)
        else:
            logger.warning(f"Unknown method {self.method}, using mutual_info")
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        self.selector.fit(X_numeric, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = [numeric_cols[i] for i, selected in enumerate(mask) if selected]
        
        logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        
        return self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to keep only selected features"""
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call fit first.")
        
        return X[self.selected_features]


class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for HOS violation prediction
    """
    
    def __init__(self, config: Optional[HOSFeatureConfig] = None):
        self.config = config or HOSFeatureConfig()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.hos_features = HOSComplianceFeatures(self.config)
        self.aggregator = AggregationFeatures()
        self.statistical_features = StatisticalFeatures()
        self.dimensionality_reducer = DimensionalityReducer()
        self.feature_selector = FeatureSelector()
        
        self.feature_names = []
        self.is_fitted = False
    
    def engineer_temporal_features(self, df: pd.DataFrame,
                                   timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Create all temporal features"""
        logger.info("Engineering temporal features...")
        
        # Cyclic features
        df = self.temporal_extractor.create_cyclic_features(df, timestamp_col)
        
        # Holiday features
        df = self.temporal_extractor.create_holiday_features(df, timestamp_col)
        
        # Time series features
        if 'driver_id' in df.columns:
            value_cols = ['hours_worked', 'breaks_taken', 'miles_driven']
            value_cols = [col for col in value_cols if col in df.columns]
            
            if value_cols:
                df = self.temporal_extractor.create_time_series_features(
                    df, timestamp_col, value_cols, group_col='driver_id'
                )
        
        return df
    
    def engineer_hos_compliance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create HOS compliance-specific features"""
        logger.info("Engineering HOS compliance features...")
        
        # Hours remaining features
        if 'hours_worked' in df.columns:
            df = self.hos_features.calculate_hours_remaining(df)
        
        # Break compliance features
        if 'breaks_taken' in df.columns and 'hours_worked' in df.columns:
            df = self.hos_features.calculate_break_compliance(df)
        
        # Weekly compliance features
        if 'driver_id' in df.columns and 'hours_worked' in df.columns:
            df = self.hos_features.calculate_weekly_compliance(df)
        
        # Violation risk features
        df = self.hos_features.create_violation_risk_features(df)
        
        return df
    
    def engineer_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features"""
        logger.info("Engineering aggregation features...")
        
        # Driver aggregations
        if 'driver_id' in df.columns:
            df = self.aggregator.create_driver_aggregations(df)
        
        # Vehicle aggregations
        if 'vehicle_id' in df.columns:
            df = self.aggregator.create_vehicle_aggregations(df)
        
        # Carrier aggregations
        if 'carrier_id' in df.columns:
            df = self.aggregator.create_carrier_aggregations(df)
        
        # Time window aggregations
        if 'timestamp' in df.columns:
            value_cols = ['hours_worked', 'breaks_taken']
            value_cols = [col for col in value_cols if col in df.columns]
            
            if value_cols and self.config.create_rolling_features:
                df = self.aggregator.create_time_window_aggregations(
                    df, 'timestamp', value_cols, self.config.rolling_windows
                )
        
        return df
    
    def engineer_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        logger.info("Engineering statistical features...")
        
        if not self.config.create_statistical_features:
            return df
        
        # Ratio features
        ratio_pairs = [
            ('hours_worked', 'breaks_taken'),
            ('miles_driven', 'hours_worked'),
        ]
        ratio_pairs = [(c1, c2) for c1, c2 in ratio_pairs 
                       if c1 in df.columns and c2 in df.columns]
        
        if ratio_pairs:
            df = self.statistical_features.create_ratio_features(df, ratio_pairs)
        
        # Binned features
        cols_to_bin = ['hours_worked', 'breaks_taken', 'experience']
        cols_to_bin = [col for col in cols_to_bin if col in df.columns]
        
        if cols_to_bin:
            df = self.statistical_features.create_binned_features(df, cols_to_bin)
        
        return df
    
    def engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        logger.info("Engineering interaction features...")
        
        # Interactions between driver experience and hours
        if 'experience' in df.columns and 'hours_worked' in df.columns:
            df['experience_x_hours'] = df['experience'] * df['hours_worked']
            df['experience_hours_ratio'] = df['experience'] / (df['hours_worked'] + 1e-6)
        
        # Interactions between risk scores
        if 'hours_worked_risk_score' in df.columns and 'break_deficit' in df.columns:
            df['composite_risk'] = df['hours_worked_risk_score'] * (df['break_deficit'] + 1)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame,
                     target_col: Optional[str] = None) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        original_features = len(df.columns)
        
        # Temporal features
        if 'timestamp' in df.columns:
            df = self.engineer_temporal_features(df)
        
        # HOS compliance features
        df = self.engineer_hos_compliance_features(df)
        
        # Aggregation features
        df = self.engineer_aggregation_features(df)
        
        # Statistical features
        df = self.engineer_statistical_features(df)
        
        # Interaction features
        df = self.engineer_interaction_features(df)
        
        # Feature selection if target provided
        if target_col and target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            self.selected_features = self.feature_selector.fit(X, y)
            logger.info(f"Selected features: {len(self.selected_features)}")
        
        final_features = len(df.columns)
        logger.info(f"Feature engineering complete: {original_features} -> {final_features} features")
        
        self.feature_names = df.columns.tolist()
        self.is_fitted = True
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted feature engineering"""
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Apply same transformations
        if 'timestamp' in df.columns:
            df = self.engineer_temporal_features(df)
        
        df = self.engineer_hos_compliance_features(df)
        df = self.engineer_aggregation_features(df)
        df = self.engineer_statistical_features(df)
        df = self.engineer_interaction_features(df)
        
        return df
    
    def get_feature_importance_report(self, X: pd.DataFrame, 
                                     y: pd.Series) -> pd.DataFrame:
        """Generate feature importance report"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a quick random forest to get feature importances
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].fillna(0)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_numeric, y)
        
        importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# Main execution
if __name__ == "__main__":
    # Example usage
    config = HOSFeatureConfig()
    engineer = AdvancedFeatureEngineer(config)
    
    # Load preprocessed data
    # df = pd.read_parquet('preprocessed_data.parquet')
    
    # Engineer features
    # engineered_df = engineer.fit_transform(df, target_col='violation_type')
    
    # Get feature importance
    # importance = engineer.get_feature_importance_report(engineered_df, df['violation_type'])
    # print(importance.head(20))