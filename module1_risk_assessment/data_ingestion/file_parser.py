"""
NSS CSV File Parser
Handles reading and parsing CSV files from NSS surveys with chunking and validation
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Iterator, Union
from pathlib import Path
import yaml
from .utils import setup_logging, validate_file_exists

class NSSCSVParser:
    """
    Parser for NSS CSV files with dynamic schema handling and chunking support
    """
    def __init__(self, config_path: str):
        """
        Initialize parser with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logging(__name__)

    def _load_config(self, config_path:str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
        
    
    def read_csv_file(self, file_path:str, file_type:str, chunksize : Optional[int] = None)-> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        """
        Read NSS CSV file with appropriate parsing parameters
        
        Args:
            file_path: Path to CSV file
            file_type: Type of file (household, person, etc.)
            chunksize: Number of rows per chunk (None for full file)
            
        Returns:
            DataFrame or iterator of DataFrames if chunking
        """

        validate_file_exists(file_path)

        # Get file-specific configuration
        file_config = self.config['file_types'].get(file_type, {})

        # Reading Parameters
        read_params = {
            'dtype' : file_config.get('dtypes', {}),
            'parse_dates' : file_config.get('date_columns',[]),
            'na_values' : self.config.get('na_values', ['', 'NA', 'NULL', 'na', 'null', ' ', '  ', '...', '-', 'NR', 'nr']
            ),
            'encoding' : self.config.get('encoding', 'utf-8'),
            'low_memory' : False
        }

        if chunksize:
            read_params['chunksize'] = chunksize
        
        try:
            self.logger.info(f"Reading {file_type} file : {file_path}")
            df_or_chunks = pd.read_csv(file_path, **read_params)

            if chunksize :
                self.logger.info(f"File loaded in chunks of {chunksize} rows")
                return df_or_chunks
            else:
                self.logger.info(f"File loaded : {len(df_or_chunks)} rows, {len(df_or_chunks)} columns")
                return df_or_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path} : {e}")
            raise


    def validate_schema(self, df: pd.DataFrame, file_type:str)-> bool:
        """
        Validate that DataFrame has expected columns for file type
        
        Args:
            df: DataFrame to validate
            file_type: Expected file type
            
        Returns:
            Boolean indicating if schema is valid
        """
        expected_columns = self.config['file_types'][file_type].get('required_columns', [])

        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            self.logger.warning(f"Missing columns in {file_type} : {missing_columns}")
            return False
        
        self.logger.info(f"Schema validation passed for {file_type}")
        return True
    

    def get_identifier_columns(self, file_type:str) -> List[str]:
        """Get identifier columns for a file type"""
        return self.config['file_types'][file_type].get('identifier_columns', [])
