"""
Code created by: Balaam Ibencho
Date:30/9/2025
Regards:Hosoptima.com

Data Connector for HOS Violation Prediction System
PostgreSQL Database and Samsara API Integration
This is our software base code that allows us to obtain data from samsara API,
Import it to our database and finally translate it to csv/parquet/excel files
which can then be used to continously train our model.
N:B. This is a Skeletonial version of the code as we are yet to receive official
Samsara API and thus it is prone to constant changes.
"""

import psycopg2
from psycopg2 import pool, extras, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_connector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'hos_violations'
    user: str = 'postgres'
    password: str = ''
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 30
    command_timeout: int = 300
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        import os
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'hos_violations'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONN', 1)),
            max_connections=int(os.getenv('DB_MAX_CONN', 10))
        )


@dataclass
class SamsaraConfig:
    """Samsara API configuration"""
    api_token: str = ''
    base_url: str = 'https://api.samsara.com'
    api_version: str = 'v1'
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 0.3
    rate_limit_per_second: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        import os
        return cls(
            api_token=os.getenv('SAMSARA_API_TOKEN', ''),
            base_url=os.getenv('SAMSARA_BASE_URL', 'https://api.samsara.com'),
            api_version=os.getenv('SAMSARA_API_VERSION', 'v1'),
            timeout=int(os.getenv('SAMSARA_TIMEOUT', 30)),
            max_retries=int(os.getenv('SAMSARA_MAX_RETRIES', 3))
        )


def retry_on_failure(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


class ConnectionPool:
    """PostgreSQL connection pool manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout
            )
            logger.info(f"Connection pool initialized: {self.config.min_connections}-{self.config.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        if self.pool:
            return self.pool.getconn()
        raise Exception("Connection pool not initialized")
    
    def return_connection(self, connection):
        """Return connection to pool"""
        if self.pool:
            self.pool.putconn(connection)
    
    def close_all_connections(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("All connections closed")


class PostgreSQLConnector:
    """
    Advanced PostgreSQL connector with connection pooling, 
    query optimization, and comprehensive error handling
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self.pool_manager = ConnectionPool(self.config)
        self.query_cache = {}
        
    @retry_on_failure(max_attempts=3, delay=1, exceptions=(psycopg2.OperationalError,))
    def execute_query(self, query: str, params: Optional[Tuple] = None, 
                     fetch: bool = True) -> Optional[List[Tuple]]:
        """Execute SQL query with retry logic"""
        connection = None
        cursor = None
        
        try:
            connection = self.pool_manager.get_connection()
            cursor = connection.cursor()
            
            # Set statement timeout
            cursor.execute(f"SET statement_timeout = {self.config.command_timeout * 1000}")
            
            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                results = cursor.fetchall()
                connection.commit()
                return results
            else:
                connection.commit()
                return None
                
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.pool_manager.return_connection(connection)
    
    def execute_query_to_dataframe(self, query: str, 
                                   params: Optional[Tuple] = None,
                                   chunksize: Optional[int] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        connection = None
        
        try:
            connection = self.pool_manager.get_connection()
            
            if chunksize:
                # For large results, use chunked reading
                chunks = []
                for chunk in pd.read_sql_query(query, connection, params=params, chunksize=chunksize):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql_query(query, connection, params=params)
            
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to execute query to DataFrame: {str(e)}")
            raise
        finally:
            if connection:
                self.pool_manager.return_connection(connection)
    
    def bulk_insert(self, table: str, data: List[Tuple], 
                   columns: Optional[List[str]] = None) -> int:
        """Bulk insert data using execute_values for performance"""
        connection = None
        cursor = None
        
        try:
            connection = self.pool_manager.get_connection()
            cursor = connection.cursor()
            
            if columns:
                column_names = ', '.join(columns)
                query = f"INSERT INTO {table} ({column_names}) VALUES %s"
            else:
                query = f"INSERT INTO {table} VALUES %s"
            
            # Use execute_values for bulk insert
            extras.execute_values(cursor, query, data)
            
            rows_inserted = cursor.rowcount
            connection.commit()
            
            logger.info(f"Bulk inserted {rows_inserted} rows into {table}")
            return rows_inserted
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Bulk insert failed: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.pool_manager.return_connection(connection)
    
    def bulk_insert_dataframe(self, table: str, df: pd.DataFrame, 
                             if_exists: str = 'append') -> int:
        """Bulk insert DataFrame to table"""
        connection = None
        
        try:
            connection = self.pool_manager.get_connection()
            
            df.to_sql(
                table,
                connection,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Inserted {len(df)} rows into {table}")
            return len(df)
            
        except Exception as e:
            logger.error(f"DataFrame bulk insert failed: {str(e)}")
            raise
        finally:
            if connection:
                self.pool_manager.return_connection(connection)
    
    def get_hos_violations(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          driver_ids: Optional[List[str]] = None,
                          violation_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get HOS violations with filters"""
        query = """
            SELECT 
                v.violation_id,
                v.driver_id,
                v.vehicle_id,
                v.violation_type,
                v.violation_code,
                v.severity,
                v.timestamp,
                v.duration_minutes,
                v.description,
                d.name as driver_name,
                d.experience_years,
                d.total_violations,
                vh.vehicle_type,
                vh.vehicle_model
            FROM hos_violations v
            LEFT JOIN drivers d ON v.driver_id = d.driver_id
            LEFT JOIN vehicles vh ON v.vehicle_id = vh.vehicle_id
            WHERE 1=1
        """
        
        params = []
        
        if start_date:
            query += " AND v.timestamp >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND v.timestamp <= %s"
            params.append(end_date)
        
        if driver_ids:
            query += " AND v.driver_id = ANY(%s)"
            params.append(driver_ids)
        
        if violation_types:
            query += " AND v.violation_type = ANY(%s)"
            params.append(violation_types)
        
        query += " ORDER BY v.timestamp DESC"
        
        return self.execute_query_to_dataframe(query, tuple(params) if params else None)
    
    def get_driver_logs(self, driver_id: str, 
                       start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """Get driver logs for a specific period"""
        query = """
            SELECT 
                log_id,
                driver_id,
                vehicle_id,
                log_date,
                start_time,
                end_time,
                hours_worked,
                breaks_taken,
                break_duration_minutes,
                miles_driven,
                duty_status,
                location_start,
                location_end,
                created_at,
                updated_at
            FROM driver_logs
            WHERE driver_id = %s
            AND log_date BETWEEN %s AND %s
            ORDER BY log_date, start_time
        """
        
        return self.execute_query_to_dataframe(query, (driver_id, start_date, end_date))
    
    def get_fleet_summary(self, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get fleet-wide summary statistics"""
        query = """
            SELECT 
                COUNT(DISTINCT d.driver_id) as total_drivers,
                COUNT(DISTINCT v.vehicle_id) as total_vehicles,
                COUNT(DISTINCT hv.violation_id) as total_violations,
                AVG(dl.hours_worked) as avg_hours_worked,
                SUM(dl.miles_driven) as total_miles_driven,
                COUNT(DISTINCT CASE WHEN hv.severity = 'HIGH' THEN hv.violation_id END) as high_severity_violations,
                COUNT(DISTINCT CASE WHEN hv.severity = 'MEDIUM' THEN hv.violation_id END) as medium_severity_violations,
                COUNT(DISTINCT CASE WHEN hv.severity = 'LOW' THEN hv.violation_id END) as low_severity_violations
            FROM drivers d
            LEFT JOIN driver_logs dl ON d.driver_id = dl.driver_id
            LEFT JOIN vehicles v ON dl.vehicle_id = v.vehicle_id
            LEFT JOIN hos_violations hv ON d.driver_id = hv.driver_id
            WHERE 1=1
        """
        
        params = []
        
        if start_date:
            query += " AND dl.log_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND dl.log_date <= %s"
            params.append(end_date)
        
        return self.execute_query_to_dataframe(query, tuple(params) if params else None)
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        tables = {
            'drivers': """
                CREATE TABLE IF NOT EXISTS drivers (
                    driver_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255),
                    phone VARCHAR(50),
                    license_number VARCHAR(100),
                    license_state VARCHAR(10),
                    experience_years INTEGER,
                    total_violations INTEGER DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'vehicles': """
                CREATE TABLE IF NOT EXISTS vehicles (
                    vehicle_id VARCHAR(50) PRIMARY KEY,
                    vehicle_number VARCHAR(100) NOT NULL,
                    vehicle_type VARCHAR(50),
                    vehicle_model VARCHAR(100),
                    vehicle_year INTEGER,
                    vin VARCHAR(50),
                    license_plate VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'carriers': """
                CREATE TABLE IF NOT EXISTS carriers (
                    carrier_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    dot_number VARCHAR(50),
                    mc_number VARCHAR(50),
                    rating DECIMAL(3,2),
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'driver_logs': """
                CREATE TABLE IF NOT EXISTS driver_logs (
                    log_id SERIAL PRIMARY KEY,
                    driver_id VARCHAR(50) REFERENCES drivers(driver_id),
                    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
                    log_date DATE NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    hours_worked DECIMAL(5,2),
                    breaks_taken INTEGER,
                    break_duration_minutes INTEGER,
                    miles_driven DECIMAL(10,2),
                    duty_status VARCHAR(20),
                    location_start VARCHAR(255),
                    location_end VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(driver_id, log_date, start_time)
                )
            """,
            'hos_violations': """
                CREATE TABLE IF NOT EXISTS hos_violations (
                    violation_id SERIAL PRIMARY KEY,
                    driver_id VARCHAR(50) REFERENCES drivers(driver_id),
                    vehicle_id VARCHAR(50) REFERENCES vehicles(vehicle_id),
                    violation_type VARCHAR(100) NOT NULL,
                    violation_code VARCHAR(50),
                    severity VARCHAR(20),
                    timestamp TIMESTAMP NOT NULL,
                    duration_minutes INTEGER,
                    description TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_driver_logs_driver_date ON driver_logs(driver_id, log_date)",
            "CREATE INDEX IF NOT EXISTS idx_violations_driver_time ON hos_violations(driver_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_violations_type ON hos_violations(violation_type)",
            "CREATE INDEX IF NOT EXISTS idx_violations_severity ON hos_violations(severity)"
        ]
        
        connection = None
        cursor = None
        
        try:
            connection = self.pool_manager.get_connection()
            connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = connection.cursor()
            
            # Create tables
            for table_name, create_sql in tables.items():
                cursor.execute(create_sql)
                logger.info(f"Table '{table_name}' created or already exists")
            
            # Create indexes
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            logger.info("All indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.pool_manager.return_connection(connection)
    
    def close(self):
        """Close all database connections"""
        self.pool_manager.close_all_connections()


class SamsaraAPIConnector:
    """
    Advanced Samsara API connector with rate limiting,
    retry logic, and comprehensive endpoint coverage
    """
    
    def __init__(self, config: Optional[SamsaraConfig] = None):
        self.config = config or SamsaraConfig.from_env()
        self.session = self._create_session()
        self.base_url = f"{self.config.base_url}/{self.config.api_version}"
        self.last_request_time = 0
        self.request_count = 0
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'Authorization': f'Bearer {self.config.api_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        return session
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < (1.0 / self.config.rate_limit_per_second):
            sleep_time = (1.0 / self.config.rate_limit_per_second) - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    @retry_on_failure(max_attempts=3, delay=1, exceptions=(requests.RequestException,))
    def _make_request(self, method: str, endpoint: str, 
                     params: Optional[Dict] = None,
                     json_data: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_data,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            
            return response.json() if response.content else {}
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(e.response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return self._make_request(method, endpoint, params, json_data)
            else:
                logger.error(f"HTTP error occurred: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_drivers(self, limit: int = 100, after: Optional[str] = None) -> Dict:
        """Get list of drivers"""
        params = {'limit': limit}
        if after:
            params['after'] = after
        
        response = self._make_request('GET', 'fleet/drivers', params=params)
        logger.info(f"Retrieved {len(response.get('data', []))} drivers")
        
        return response
    
    def get_driver_hos_logs(self, driver_id: str, 
                           start_time: datetime,
                           end_time: datetime) -> Dict:
        """Get HOS logs for a specific driver"""
        params = {
            'startMs': int(start_time.timestamp() * 1000),
            'endMs': int(end_time.timestamp() * 1000)
        }
        
        endpoint = f'fleet/drivers/{driver_id}/hos_daily_logs'
        response = self._make_request('GET', endpoint, params=params)
        
        logger.info(f"Retrieved HOS logs for driver {driver_id}")
        
        return response
    
    def get_hos_logs_batch(self, driver_ids: List[str],
                          start_time: datetime,
                          end_time: datetime) -> List[Dict]:
        """Get HOS logs for multiple drivers in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    self.get_driver_hos_logs,
                    driver_id,
                    start_time,
                    end_time
                ): driver_id
                for driver_id in driver_ids
            }
            
            for future in as_completed(futures):
                driver_id = futures[future]
                try:
                    data = future.result()
                    results.append({
                        'driver_id': driver_id,
                        'data': data,
                        'status': 'success'
                    })
                except Exception as e:
                    logger.error(f"Failed to get logs for driver {driver_id}: {str(e)}")
                    results.append({
                        'driver_id': driver_id,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return results
    
    def get_vehicles(self, limit: int = 100, after: Optional[str] = None) -> Dict:
        """Get list of vehicles"""
        params = {'limit': limit}
        if after:
            params['after'] = after
        
        response = self._make_request('GET', 'fleet/vehicles', params=params)
        logger.info(f"Retrieved {len(response.get('data', []))} vehicles")
        
        return response
    
    def get_vehicle_locations(self, start_time: datetime, 
                             end_time: datetime) -> Dict:
        """Get vehicle locations"""
        params = {
            'startMs': int(start_time.timestamp() * 1000),
            'endMs': int(end_time.timestamp() * 1000)
        }
        
        response = self._make_request('GET', 'fleet/vehicles/locations', params=params)
        logger.info(f"Retrieved locations for {len(response.get('data', []))} vehicles")
        
        return response
    
    def get_driver_safety_scores(self, start_time: datetime,
                                 end_time: datetime) -> Dict:
        """Get driver safety scores"""
        params = {
            'startMs': int(start_time.timestamp() * 1000),
            'endMs': int(end_time.timestamp() * 1000)
        }
        
        response = self._make_request('GET', 'fleet/drivers/safety/scores', params=params)
        logger.info("Retrieved driver safety scores")
        
        return response
    
    def get_hos_violations(self, start_time: datetime,
                          end_time: datetime) -> Dict:
        """Get HOS violations"""
        params = {
            'startMs': int(start_time.timestamp() * 1000),
            'endMs': int(end_time.timestamp() * 1000)
        }
        
        response = self._make_request('GET', 'fleet/hos/violations', params=params)
        logger.info(f"Retrieved {len(response.get('data', []))} HOS violations")
        
        return response
    
    def get_driver_stats(self, driver_id: str,
                        start_time: datetime,
                        end_time: datetime) -> Dict:
        """Get comprehensive driver statistics"""
        params = {
            'startMs': int(start_time.timestamp() * 1000),
            'endMs': int(end_time.timestamp() * 1000)
        }
        
        endpoint = f'fleet/drivers/{driver_id}/stats'
        response = self._make_request('GET', endpoint, params=params)
        
        logger.info(f"Retrieved stats for driver {driver_id}")
        
        return response
    
    def sync_to_dataframe(self, data_type: str, 
                         start_time: datetime,
                         end_time: datetime,
                         driver_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Sync Samsara data to DataFrame"""
        if data_type == 'hos_logs' and driver_ids:
            results = self.get_hos_logs_batch(driver_ids, start_time, end_time)
            # Flatten and convert to DataFrame
            rows = []
            for result in results:
                if result['status'] == 'success':
                    logs = result['data'].get('data', [])
                    for log in logs:
                        log['driver_id'] = result['driver_id']
                        rows.append(log)
            
            return pd.DataFrame(rows)
        
        elif data_type == 'violations':
            response = self.get_hos_violations(start_time, end_time)
            return pd.DataFrame(response.get('data', []))
        
        elif data_type == 'vehicles':
            response = self.get_vehicles()
            return pd.DataFrame(response.get('data', []))
        
        elif data_type == 'drivers':
            response = self.get_drivers()
            return pd.DataFrame(response.get('data', []))
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


class IntegratedDataConnector:
    """
    Integrated connector combining PostgreSQL and Samsara API
    with automatic synchronization and data pipeline management
    """
    
    def __init__(self, 
                 db_config: Optional[DatabaseConfig] = None,
                 samsara_config: Optional[SamsaraConfig] = None):
        self.db_connector = PostgreSQLConnector(db_config)
        self.samsara_connector = SamsaraAPIConnector(samsara_config)
        
    def sync_samsara_to_database(self, data_type: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 driver_ids: Optional[List[str]] = None) -> int:
        """Sync data from Samsara API to PostgreSQL database"""
        logger.info(f"Syncing {data_type} from Samsara to database...")
        
        try:
            # Get data from Samsara
            df = self.samsara_connector.sync_to_dataframe(
                data_type, start_time, end_time, driver_ids
            )
            
            if df.empty:
                logger.warning(f"No {data_type} data retrieved from Samsara")
                return 0
            
            # Map data types to tables
            table_mapping = {
                'drivers': 'drivers',
                'vehicles': 'vehicles',
                'hos_logs': 'driver_logs',
                'violations': 'hos_violations'
            }
            
            table_name = table_mapping.get(data_type)
            if not table_name:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Insert into database
            rows_inserted = self.db_connector.bulk_insert_dataframe(
                table_name, df, if_exists='append'
            )
            
            logger.info(f"Successfully synced {rows_inserted} {data_type} records to database")
            
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            raise
    
    def get_combined_driver_data(self, driver_id: str,
                                start_date: datetime,
                                end_date: datetime,
                                include_samsara: bool = True) -> pd.DataFrame:
        """Get combined driver data from database and optionally Samsara"""
        # Get data from database
        db_data = self.db_connector.get_driver_logs(driver_id, start_date, end_date)
        
        if include_samsara:
            try:
                # Get latest data from Samsara
                samsara_data = self.samsara_connector.get_driver_hos_logs(
                    driver_id, start_date, end_date
                )
                
                # Convert and merge
                samsara_df = pd.DataFrame(samsara_data.get('data', []))
                
                if not samsara_df.empty:
                    # Merge datasets
                    combined_data = pd.concat([db_data, samsara_df], ignore_index=True)
                    # Remove duplicates
                    combined_data = combined_data.drop_duplicates()
                    
                    logger.info(f"Combined data: {len(db_data)} from DB + {len(samsara_df)} from Samsara")
                    return combined_data
                
            except Exception as e:
                logger.warning(f"Failed to get Samsara data: {str(e)}. Using DB data only.")
        
        return db_data
    
    def get_real_time_fleet_data(self, lookback_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """Get real-time fleet data from both sources"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        data = {}
        
        # Get drivers
        try:
            drivers_api = self.samsara_connector.get_drivers()
            data['drivers'] = pd.DataFrame(drivers_api.get('data', []))
        except Exception as e:
            logger.error(f"Failed to get drivers from API: {str(e)}")
            data['drivers'] = pd.DataFrame()
        
        # Get vehicles
        try:
            vehicles_api = self.samsara_connector.get_vehicles()
            data['vehicles'] = pd.DataFrame(vehicles_api.get('data', []))
        except Exception as e:
            logger.error(f"Failed to get vehicles from API: {str(e)}")
            data['vehicles'] = pd.DataFrame()
        
        # Get recent violations
        try:
            violations_api = self.samsara_connector.get_hos_violations(start_time, end_time)
            data['violations'] = pd.DataFrame(violations_api.get('data', []))
        except Exception as e:
            logger.error(f"Failed to get violations from API: {str(e)}")
            # Fallback to database
            data['violations'] = self.db_connector.get_hos_violations(start_time, end_time)
        
        # Get fleet summary from database
        try:
            data['fleet_summary'] = self.db_connector.get_fleet_summary(start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to get fleet summary: {str(e)}")
            data['fleet_summary'] = pd.DataFrame()
        
        logger.info("Retrieved real-time fleet data")
        
        return data
    
    def automated_sync_pipeline(self, sync_interval_hours: int = 1,
                               lookback_hours: int = 24) -> Dict[str, int]:
        """Automated sync pipeline for continuous data synchronization"""
        logger.info("Starting automated sync pipeline...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        results = {}
        
        # Sync drivers
        try:
            driver_rows = self.sync_samsara_to_database('drivers', start_time, end_time)
            results['drivers'] = driver_rows
        except Exception as e:
            logger.error(f"Failed to sync drivers: {str(e)}")
            results['drivers'] = 0
        
        # Sync vehicles
        try:
            vehicle_rows = self.sync_samsara_to_database('vehicles', start_time, end_time)
            results['vehicles'] = vehicle_rows
        except Exception as e:
            logger.error(f"Failed to sync vehicles: {str(e)}")
            results['vehicles'] = 0
        
        # Sync violations
        try:
            violation_rows = self.sync_samsara_to_database('violations', start_time, end_time)
            results['violations'] = violation_rows
        except Exception as e:
            logger.error(f"Failed to sync violations: {str(e)}")
            results['violations'] = 0
        
        # Get list of drivers for HOS logs
        try:
            drivers_response = self.samsara_connector.get_drivers()
            driver_ids = [d.get('id') for d in drivers_response.get('data', [])]
            
            if driver_ids:
                log_rows = self.sync_samsara_to_database('hos_logs', start_time, end_time, driver_ids)
                results['hos_logs'] = log_rows
        except Exception as e:
            logger.error(f"Failed to sync HOS logs: {str(e)}")
            results['hos_logs'] = 0
        
        total_rows = sum(results.values())
        logger.info(f"Sync pipeline completed: {total_rows} total rows synced")
        logger.info(f"Breakdown: {results}")
        
        return results
    
    def get_training_dataset(self, start_date: datetime,
                           end_date: datetime,
                           include_features: bool = True) -> pd.DataFrame:
        """Get complete training dataset with all features"""
        logger.info("Building training dataset...")
        
        # Get base HOS violations
        violations = self.db_connector.get_hos_violations(start_date, end_date)
        
        if violations.empty:
            logger.warning("No violations found in date range")
            return pd.DataFrame()
        
        # Get driver logs for context
        driver_ids = violations['driver_id'].unique().tolist()
        
        all_logs = []
        for driver_id in driver_ids:
            try:
                logs = self.db_connector.get_driver_logs(driver_id, start_date, end_date)
                all_logs.append(logs)
            except Exception as e:
                logger.warning(f"Failed to get logs for driver {driver_id}: {str(e)}")
        
        if all_logs:
            logs_df = pd.concat(all_logs, ignore_index=True)
            
            # Merge violations with logs
            dataset = violations.merge(
                logs_df,
                on=['driver_id', 'vehicle_id'],
                how='left',
                suffixes=('_violation', '_log')
            )
        else:
            dataset = violations
        
        logger.info(f"Training dataset created with {len(dataset)} records")
        
        return dataset
    
    def export_to_files(self, output_dir: str, 
                       start_date: datetime,
                       end_date: datetime,
                       file_format: str = 'parquet'):
        """Export database data to files for backup or analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export violations
        violations = self.db_connector.get_hos_violations(start_date, end_date)
        if not violations.empty:
            if file_format == 'parquet':
                violations.to_parquet(output_path / f'hos_violations_{timestamp}.parquet', index=False)
            elif file_format == 'csv':
                violations.to_csv(output_path / f'hos_violations_{timestamp}.csv', index=False)
            logger.info(f"Exported {len(violations)} violations")
        
        # Export fleet summary
        fleet_summary = self.db_connector.get_fleet_summary(start_date, end_date)
        if not fleet_summary.empty:
            if file_format == 'parquet':
                fleet_summary.to_parquet(output_path / f'fleet_summary_{timestamp}.parquet', index=False)
            elif file_format == 'csv':
                fleet_summary.to_csv(output_path / f'fleet_summary_{timestamp}.csv', index=False)
            logger.info("Exported fleet summary")
        
        logger.info(f"Export completed to {output_dir}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all connections"""
        health = {
            'database': False,
            'samsara_api': False,
            'overall': False
        }
        
        # Check database
        try:
            test_query = "SELECT 1"
            self.db_connector.execute_query(test_query)
            health['database'] = True
            logger.info("Database connection: OK")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
        
        # Check Samsara API
        try:
            self.samsara_connector.get_drivers(limit=1)
            health['samsara_api'] = True
            logger.info("Samsara API connection: OK")
        except Exception as e:
            logger.error(f"Samsara API connection failed: {str(e)}")
        
        health['overall'] = health['database'] and health['samsara_api']
        
        return health
    
    def get_statistics(self) -> Dict:
        """Get statistics about data connector usage"""
        stats = {
            'database': {
                'pool_size': self.db_connector.config.max_connections,
                'min_connections': self.db_connector.config.min_connections
            },
            'samsara': {
                'total_requests': self.samsara_connector.request_count,
                'rate_limit': self.samsara_connector.config.rate_limit_per_second
            }
        }
        
        return stats
    
    def close(self):
        """Close all connections"""
        logger.info("Closing all connections...")
        self.db_connector.close()
        logger.info("All connections closed")


class DataConnectorFactory:
    """Factory for creating data connectors with different configurations"""
    
    @staticmethod
    def create_from_config_file(config_path: str) -> IntegratedDataConnector:
        """Create connector from configuration file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
        
        db_config = DatabaseConfig.from_dict(config.get('database', {}))
        samsara_config = SamsaraConfig.from_dict(config.get('samsara', {}))
        
        return IntegratedDataConnector(db_config, samsara_config)
    
    @staticmethod
    def create_from_env() -> IntegratedDataConnector:
        """Create connector from environment variables"""
        db_config = DatabaseConfig.from_env()
        samsara_config = SamsaraConfig.from_env()
        
        return IntegratedDataConnector(db_config, samsara_config)
    
    @staticmethod
    def create_with_defaults() -> IntegratedDataConnector:
        """Create connector with default configuration"""
        return IntegratedDataConnector()


# Scheduled sync job example
class ScheduledSyncJob:
    """Scheduled job for continuous data synchronization"""
    
    def __init__(self, connector: IntegratedDataConnector, 
                 interval_minutes: int = 60,
                 lookback_hours: int = 24):
        self.connector = connector
        self.interval_minutes = interval_minutes
        self.lookback_hours = lookback_hours
        self.running = False
        self.last_sync_time = None
        self.sync_history = []
    
    def start(self):
        """Start scheduled sync job"""
        import threading
        
        self.running = True
        logger.info(f"Starting scheduled sync job (interval: {self.interval_minutes} min)")
        
        def run_sync():
            while self.running:
                try:
                    logger.info("Running scheduled sync...")
                    results = self.connector.automated_sync_pipeline(
                        sync_interval_hours=self.interval_minutes / 60,
                        lookback_hours=self.lookback_hours
                    )
                    
                    self.last_sync_time = datetime.now()
                    self.sync_history.append({
                        'timestamp': self.last_sync_time,
                        'results': results,
                        'status': 'success'
                    })
                    
                    # Keep only last 100 sync records
                    if len(self.sync_history) > 100:
                        self.sync_history = self.sync_history[-100:]
                    
                except Exception as e:
                    logger.error(f"Scheduled sync failed: {str(e)}")
                    self.sync_history.append({
                        'timestamp': datetime.now(),
                        'error': str(e),
                        'status': 'failed'
                    })
                
                # Wait for next interval
                time.sleep(self.interval_minutes * 60)
        
        sync_thread = threading.Thread(target=run_sync, daemon=True)
        sync_thread.start()
        
        return sync_thread
    
    def stop(self):
        """Stop scheduled sync job"""
        self.running = False
        logger.info("Scheduled sync job stopped")
    
    def get_sync_history(self, last_n: int = 10) -> List[Dict]:
        """Get last n sync results"""
        return self.sync_history[-last_n:]


# Main execution and examples
if __name__ == "__main__":
    # Example 1: Create connector from environment variables
    logger.info("Example 1: Creating connector from environment")
    connector = DataConnectorFactory.create_from_env()
    
    # Example 2: Health check
    logger.info("\nExample 2: Health check")
    health = connector.health_check()
    print(f"Health Status: {json.dumps(health, indent=2)}")
    
    # Example 3: Create database tables
    logger.info("\nExample 3: Creating database tables")
    connector.db_connector.create_tables()
    
    # Example 4: Sync data from Samsara
    logger.info("\nExample 4: Syncing data from Samsara API")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    try:
        results = connector.automated_sync_pipeline(
            sync_interval_hours=1,
            lookback_hours=168  # 7 days
        )
        print(f"Sync Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
    
    # Example 5: Get real-time fleet data
    logger.info("\nExample 5: Getting real-time fleet data")
    try:
        fleet_data = connector.get_real_time_fleet_data(lookback_hours=24)
        print(f"Fleet Data Retrieved:")
        for key, df in fleet_data.items():
            print(f"  {key}: {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to get fleet data: {str(e)}")
    
    # Example 6: Get training dataset
    logger.info("\nExample 6: Building training dataset")
    try:
        dataset = connector.get_training_dataset(start_time, end_time)
        print(f"Training dataset created with {len(dataset)} records")
        if not dataset.empty:
            print(f"Columns: {dataset.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to build dataset: {str(e)}")
    
    # Example 7: Get specific driver data
    logger.info("\nExample 7: Getting driver-specific data")
    try:
        # Replace with actual driver ID
        driver_data = connector.get_combined_driver_data(
            driver_id="driver_123",
            start_date=start_time,
            end_date=end_time,
            include_samsara=True
        )
        print(f"Retrieved {len(driver_data)} records for driver")
    except Exception as e:
        logger.error(f"Failed to get driver data: {str(e)}")
    
    # Example 8: Export data to files
    logger.info("\nExample 8: Exporting data to files")
    try:
        connector.export_to_files(
            output_dir='./data_exports',
            start_date=start_time,
            end_date=end_time,
            file_format='parquet'
        )
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
    
    # Example 9: Start scheduled sync job
    logger.info("\nExample 9: Starting scheduled sync job")
    try:
        sync_job = ScheduledSyncJob(
            connector,
            interval_minutes=60,
            lookback_hours=24
        )
        # sync_thread = sync_job.start()
        # To stop: sync_job.stop()
        logger.info("Scheduled sync job configured (not started in example)")
    except Exception as e:
        logger.error(f"Failed to configure sync job: {str(e)}")
    
    # Example 10: Get connector statistics
    logger.info("\nExample 10: Getting connector statistics")
    stats = connector.get_statistics()
    print(f"Connector Statistics: {json.dumps(stats, indent=2)}")
    
    # Example 11: Query examples for PostgreSQL
    logger.info("\nExample 11: Custom query examples")
    try:
        # Get high severity violations
        high_severity_query = """
            SELECT 
                v.violation_id,
                v.driver_id,
                d.name,
                v.violation_type,
                v.severity,
                v.timestamp
            FROM hos_violations v
            JOIN drivers d ON v.driver_id = d.driver_id
            WHERE v.severity = 'HIGH'
            AND v.timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY v.timestamp DESC
            LIMIT 100
        """
        
        high_severity_df = connector.db_connector.execute_query_to_dataframe(high_severity_query)
        print(f"Found {len(high_severity_df)} high severity violations")
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
    
    # Example 12: Bulk insert example
    logger.info("\nExample 12: Bulk insert example")
    try:
        # Example bulk data
        sample_data = [
            ('driver_001', 'John Doe', 'john@example.com', '555-1234', 5, 0),
            ('driver_002', 'Jane Smith', 'jane@example.com', '555-5678', 3, 2),
        ]
        
        # Would insert like this:
        # rows = connector.db_connector.bulk_insert(
        #     table='drivers',
        #     data=sample_data,
        #     columns=['driver_id', 'name', 'email', 'phone', 'experience_years', 'total_violations']
        # )
        # print(f"Inserted {rows} rows")
        
        logger.info("Bulk insert example configured (not executed)")
    except Exception as e:
        logger.error(f"Bulk insert failed: {str(e)}")
    
    # Cleanup
    logger.info("\nClosing connections...")
    connector.close()
    logger.info("Example script completed")
    
    # Configuration file example
    logger.info("\n" + "="*80)
    logger.info("CONFIGURATION FILE EXAMPLE (config.yaml):")
    logger.info("="*80)
    
    example_config = """
database:
  host: localhost
  port: 5432
  database: hos_violations
  user: postgres
  password: your_password
  min_connections: 2
  max_connections: 20
  connection_timeout: 30
  command_timeout: 300

samsara:
  api_token: your_samsara_api_token_here
  base_url: https://api.samsara.com
  api_version: v1
  timeout: 30
  max_retries: 3
  retry_backoff: 0.3
  rate_limit_per_second: 10
"""
    print(example_config)
    