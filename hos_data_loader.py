"""
Code created by: Balaam Ibencho
Date:22/9/2025
Regards:Hosoptima.com

Advanced Data Loader for HOS Violation Prediction System
Handles loading, merging, and initial validation of multiple data sources
Production-ready with comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoaderConfig:
    """Configuration class for data loading parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_path = self.config.get('data_path', './data')
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.parallel_loading = self.config.get('parallel_loading', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.date_columns = self.config.get('date_columns', [])
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
        return {}


class DataValidator:
    """Validates loaded data for integrity and completeness"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                          df_name: str) -> Tuple[bool, List[str]]:
        """Validate dataframe structure and required columns"""
        issues = []
        
        if df is None or df.empty:
            issues.append(f"{df_name}: DataFrame is empty or None")
            return False, issues
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"{df_name}: Missing columns: {missing_cols}")
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(f"{df_name}: Found {dup_count} duplicate indices")
        
        # Check data types
        for col in df.columns:
            if df[col].dtype == object:
                null_count = df[col].isnull().sum()
                if null_count > len(df) * 0.5:
                    issues.append(f"{df_name}: Column '{col}' has >50% null values")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_referential_integrity(main_df: pd.DataFrame, 
                                      ref_df: pd.DataFrame,
                                      main_key: str, 
                                      ref_key: str) -> Tuple[bool, str]:
        """Check referential integrity between dataframes"""
        main_keys = set(main_df[main_key].unique())
        ref_keys = set(ref_df[ref_key].unique())
        
        missing_refs = main_keys - ref_keys
        if missing_refs:
            return False, f"Found {len(missing_refs)} orphaned references"
        
        return True, "Referential integrity maintained"


class DataCache:
    """Simple caching mechanism for loaded data"""
    
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, filepath: str) -> str:
        """Generate cache key based on file path and modification time"""
        file_path = Path(filepath)
        if not file_path.exists():
            return None
        
        mod_time = file_path.stat().st_mtime
        key_string = f"{filepath}_{mod_time}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, filepath: str) -> Optional[pd.DataFrame]:
        """Retrieve cached dataframe if available"""
        cache_key = self._get_cache_key(filepath)
        if not cache_key:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                logger.info(f"Loading from cache: {filepath}")
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
                return None
        return None
    
    def set(self, filepath: str, df: pd.DataFrame):
        """Cache dataframe"""
        cache_key = self._get_cache_key(filepath)
        if not cache_key:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            df.to_pickle(cache_file)
            logger.info(f"Cached data for: {filepath}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


class AdvancedDataLoader:
    """
    Advanced data loader with support for:
    - Multiple file formats (CSV, Parquet, Excel)
    - Parallel loading
    - Data validation
    - Caching
    - Automatic type inference
    - Memory optimization
    """
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        self.config = config or DataLoaderConfig()
        self.validator = DataValidator()
        self.cache = DataCache() if self.config.cache_enabled else None
        self.loaded_data = {}
        
    def _detect_file_format(self, filepath: str) -> str:
        """Detect file format from extension"""
        suffix = Path(filepath).suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.feather': 'feather'
        }
        return format_map.get(suffix, 'csv')
    
    def _load_single_file(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a single file with automatic format detection"""
        file_path = Path(self.config.data_path) / filepath
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check cache first
        if self.cache:
            cached_df = self.cache.get(str(file_path))
            if cached_df is not None:
                return cached_df
        
        file_format = self._detect_file_format(str(file_path))
        logger.info(f"Loading {file_format} file: {file_path}")
        
        try:
            if file_format == 'csv':
                df = pd.read_csv(
                    file_path,
                    encoding=self.config.encoding,
                    low_memory=False,
                    **kwargs
                )
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_format == 'excel':
                df = pd.read_excel(file_path, **kwargs)
            elif file_format == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_format == 'feather':
                df = pd.read_feather(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Cache the loaded data
            if self.cache:
                self.cache.set(str(file_path), df)
            
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage by downcasting numeric types"""
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                if str(col_type)[:3] == 'int':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif str(col_type)[:5] == 'float':
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory optimization: {original_memory:.2f}MB -> {final_memory:.2f}MB")
        
        return df
    
    def load_hos_violations(self, filepath: str = 'hos_violations.csv') -> pd.DataFrame:
        """Load HOS violations data with validation"""
        logger.info("Loading HOS violations data...")
        
        df = self._load_single_file(filepath)
        
        # Expected columns
        required_cols = ['ViolationCode', 'Description', 'Violations', 
                        'DOS Violation', 'Severity', 'weight']
        
        # Validate
        is_valid, issues = self.validator.validate_dataframe(
            df, required_cols, 'HOS Violations'
        )
        
        if not is_valid:
            logger.warning(f"Validation issues: {issues}")
        
        # Type conversions
        if 'Severity' in df.columns:
            df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
        
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        
        df = self._optimize_dtypes(df)
        self.loaded_data['hos_violations'] = df
        
        return df
    
    def load_driver_data(self, filepath: str = 'driver_data.csv') -> pd.DataFrame:
        """Load driver data with validation"""
        logger.info("Loading driver data...")
        
        df = self._load_single_file(filepath)
        
        required_cols = ['driver_id', 'experience']
        is_valid, issues = self.validator.validate_dataframe(
            df, required_cols, 'Driver Data'
        )
        
        if not is_valid:
            logger.warning(f"Validation issues: {issues}")
        
        # Convert experience to numeric
        if 'experience' in df.columns:
            df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
        
        # Handle categorical data
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'driver_id' and df[col].nunique() < 50:
                df[col] = df[col].astype('category')
        
        df = self._optimize_dtypes(df)
        self.loaded_data['driver_data'] = df
        
        return df
    
    def load_vehicle_data(self, filepath: str = 'vehicle_data.csv') -> pd.DataFrame:
        """Load vehicle data with validation"""
        logger.info("Loading vehicle data...")
        
        df = self._load_single_file(filepath)
        
        required_cols = ['vehicle_id', 'type']
        is_valid, issues = self.validator.validate_dataframe(
            df, required_cols, 'Vehicle Data'
        )
        
        if not is_valid:
            logger.warning(f"Validation issues: {issues}")
        
        # Optimize categorical columns
        if 'type' in df.columns:
            df['type'] = df['type'].astype('category')
        
        df = self._optimize_dtypes(df)
        self.loaded_data['vehicle_data'] = df
        
        return df
    
    def load_carrier_data(self, filepath: str = 'carrier_data.csv') -> pd.DataFrame:
        """Load carrier data with validation"""
        logger.info("Loading carrier data...")
        
        df = self._load_single_file(filepath)
        
        required_cols = ['carrier_id', 'name']
        is_valid, issues = self.validator.validate_dataframe(
            df, required_cols, 'Carrier Data'
        )
        
        if not is_valid:
            logger.warning(f"Validation issues: {issues}")
        
        # Convert ratings to numeric
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        df = self._optimize_dtypes(df)
        self.loaded_data['carrier_data'] = df
        
        return df
    
    def load_log_data(self, filepath: str = 'log_data.csv', 
                      chunksize: Optional[int] = None) -> pd.DataFrame:
        """Load log data (potentially large) with chunked processing"""
        logger.info("Loading log data...")
        
        chunk_size = chunksize or self.config.chunk_size
        file_path = Path(self.config.data_path) / filepath
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # For large files, use chunked reading
        chunks = []
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, 
                                    low_memory=False, encoding=self.config.encoding):
                # Process dates if present
                date_cols = ['timestamp', 'date', 'log_date', 'start_time', 'end_time']
                for col in date_cols:
                    if col in chunk.columns:
                        chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
                
                # Convert numeric columns
                numeric_cols = ['hours_worked', 'breaks_taken', 'miles_driven']
                for col in numeric_cols:
                    if col in chunk.columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
                chunks.append(chunk)
                logger.info(f"Processed chunk with {len(chunk)} rows")
        
        except Exception as e:
            logger.error(f"Error reading log data: {str(e)}")
            raise
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded total {len(df)} log records")
        
        df = self._optimize_dtypes(df)
        self.loaded_data['log_data'] = df
        
        return df
    
    def load_all_data(self, parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Load all datasets with optional parallel processing"""
        logger.info("Loading all datasets...")
        
        datasets = {
            'hos_violations': 'hos_violations.csv',
            'driver_data': 'driver_data.csv',
            'vehicle_data': 'vehicle_data.csv',
            'carrier_data': 'carrier_data.csv',
            'log_data': 'log_data.csv'
        }
        
        if parallel and self.config.parallel_loading:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                
                for name, filepath in datasets.items():
                    if name == 'hos_violations':
                        future = executor.submit(self.load_hos_violations, filepath)
                    elif name == 'driver_data':
                        future = executor.submit(self.load_driver_data, filepath)
                    elif name == 'vehicle_data':
                        future = executor.submit(self.load_vehicle_data, filepath)
                    elif name == 'carrier_data':
                        future = executor.submit(self.load_carrier_data, filepath)
                    elif name == 'log_data':
                        future = executor.submit(self.load_log_data, filepath)
                    
                    futures[future] = name
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        df = future.result()
                        logger.info(f"Successfully loaded {name}")
                    except Exception as e:
                        logger.error(f"Failed to load {name}: {str(e)}")
        else:
            # Sequential loading
            try:
                self.load_hos_violations()
                self.load_driver_data()
                self.load_vehicle_data()
                self.load_carrier_data()
                self.load_log_data()
            except Exception as e:
                logger.error(f"Error during sequential loading: {str(e)}")
                raise
        
        return self.loaded_data
    
    def merge_datasets(self, 
                      on_keys: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Merge all loaded datasets into a single dataframe"""
        logger.info("Merging datasets...")
        
        if not self.loaded_data:
            raise ValueError("No data loaded. Call load_all_data() first.")
        
        # Default merge keys
        if on_keys is None:
            on_keys = {
                'driver_data': 'driver_id',
                'vehicle_data': 'vehicle_id',
                'carrier_data': 'carrier_id'
            }
        
        # Start with log data as base
        merged_df = self.loaded_data.get('log_data')
        
        if merged_df is None:
            raise ValueError("Log data not found")
        
        # Merge driver data
        if 'driver_data' in self.loaded_data:
            driver_key = on_keys.get('driver_data', 'driver_id')
            if driver_key in merged_df.columns:
                merged_df = merged_df.merge(
                    self.loaded_data['driver_data'],
                    on=driver_key,
                    how='left',
                    suffixes=('', '_driver')
                )
                logger.info("Merged driver data")
        
        # Merge vehicle data
        if 'vehicle_data' in self.loaded_data:
            vehicle_key = on_keys.get('vehicle_data', 'vehicle_id')
            if vehicle_key in merged_df.columns:
                merged_df = merged_df.merge(
                    self.loaded_data['vehicle_data'],
                    on=vehicle_key,
                    how='left',
                    suffixes=('', '_vehicle')
                )
                logger.info("Merged vehicle data")
        
        # Merge carrier data
        if 'carrier_data' in self.loaded_data:
            carrier_key = on_keys.get('carrier_data', 'carrier_id')
            if carrier_key in merged_df.columns:
                merged_df = merged_df.merge(
                    self.loaded_data['carrier_data'],
                    on=carrier_key,
                    how='left',
                    suffixes=('', '_carrier')
                )
                logger.info("Merged carrier data")
        
        logger.info(f"Final merged dataset shape: {merged_df.shape}")
        
        return merged_df
    
    def get_data_summary(self) -> Dict:
        """Generate summary statistics for loaded data"""
        summary = {}
        
        for name, df in self.loaded_data.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'null_counts': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
        
        return summary
    
    def export_merged_data(self, output_path: str, format: str = 'parquet'):
        """Export merged data to file"""
        if not self.loaded_data:
            raise ValueError("No data to export")
        
        merged_df = self.merge_datasets()
        
        if format == 'parquet':
            merged_df.to_parquet(output_path, index=False)
        elif format == 'csv':
            merged_df.to_csv(output_path, index=False)
        elif format == 'feather':
            merged_df.to_feather(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported merged data to {output_path}")


# Main execution
if __name__ == "__main__":
    # Example usage
    config = DataLoaderConfig()
    loader = AdvancedDataLoader(config)
    
    # Load all data
    data = loader.load_all_data(parallel=True)
    
    # Get summary
    summary = loader.get_data_summary()
    print(json.dumps(summary, indent=2))
    
    # Merge datasets
    merged_data = loader.merge_datasets()
    print(f"\nMerged dataset shape: {merged_data.shape}")
    
    # Export
    # loader.export_merged_data('merged_hos_data.parquet')
