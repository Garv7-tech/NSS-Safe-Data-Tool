"""
NSS Data Cleaning and Preprocessing
Handles type conversion, null handling, and standardization
"""
import zipfile
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .utils import setup_logging


class NSSDataCleaner:
    """
    Data cleaner for NSS datasets with survey-specific cleaning rules
    """

    def __init__(self, config : Dict):
        """
        Initialize cleaner with configuration
        
        Args:
            config: Configuration dictionary
        """

        self.config = config
        self.logger = setup_logging(__name__)


    def unzip_file(self, zip_path: str, extract_to: Optional[str] = None) -> None:
        """
        Unzip a ZIP file to a specified directory or to the current directory.
        
        Args:
            zip_path: Path to the ZIP file.
            extract_to: Directory to extract contents to. Defaults to current directory.
        """
        if extract_to is None:
            extract_to = os.getcwd()
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            self.logger.info(f"Extracted all files to: {extract_to}")
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path}: {e}")


    def clean_dataframe(self, df: pd.DataFrame, file_type: str)-> pd.DataFrame :
        """
        Apply cleaning rules to DataFrame
        
        Args:
            df: Input DataFrame
            file_type: Type of file being cleaned
            
        Returns:
            Cleaned DataFrame
        """

        df_cleaned = df.copy()

        # Apply cleaning steps
        df_cleaned = self._standardize_column_names(df_cleaned, file_type)
        df_cleaned = self._convert_data_types(df_cleaned, file_type)
        df_cleaned = self._handle_missing_values(df_cleaned, file_type)
        df_cleaned = self.standardize_categorical_values(df_cleaned, file_type)
        df_cleaned = self._validate_ranges(df_cleaned, file_type)

        self.logger.info(f"Data cleaning completed for {file_type}")
        return df_cleaned
    
    def _standardize_column_names(self, df:pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Standardize column names based on configuration"""
        column_mapping = self.config['file_types'][file_type].get('column_mapping', {})

        if column_mapping:
            df = df.rename(columns = column_mapping)
            self.logger.info(f"Column names standardized for {file_type}")

        return df
    

    def _convert_data_types(self,df:pd.DataFrame, file_type:str)->pd.DataFrame:
        """Convert data types according to configuration"""

        dtypes = self.config['file_types'][file_type].get('dtypes',{})

        for column , dtype in dtypes.items():
            if column in df.columns:
                try:
                    if dtype == 'category':
                        df[column] = df[column].astype('category')
                    elif dtype == 'datetime':
                        df[column] = pd.to_datetime(df[column],errors= 'coerce')
                except Exception as e:
                    self.logger.warning(f"Failed to convert {column} to {dtype} : {e}")

        return df
    
    def _handle_missing_values(self, df:pd.DataFrame, file_type:str)->pd.DataFrame:
        """Handle missing values according to configuration"""

        missing_rules = self.config['file_types'][file_type].get('missing_value_rules', {})

        for column, rule in missing_rules.items():
            if column in df.columns:
                if rule == 'drop_rows':
                    df = df.dropna(subset=[column])
                elif rule =='fill_zero':
                    df = df.fillna(0)

                elif rule == 'fill_mode':
                    mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 0 # df[column].mode() return a Series

                    df[column] = df[column].fillna(mode_value)

        return df
    

    def _standardized_categorical_values(self,df:pd.DataFrame, file_type:str)->pd.DataFrame:
        """Standardize categorical values using codebooks"""

        categorical_mappings = self.config['file_types'][file_type].get('categorical_mappings', {})

        for column, mapping in categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping).fillna(df[column])

        return df
    
    def _validate_ranges(self, df:pd.DataFrame, file_type : str)->pd.DataFrame:
        """Validate numerical ranges and flag outliers"""

        range_rules = self.config['file_types'][file_type].get('range_validation',{})

        for column, rules in range_rules.items():
            if column in df.columns:
                min_val = rules.get('min')
                max_val = rules.get('max')

                if min_val is not None : 
                    outliers = df[df[column]< min_val ].shape(0)
                    # .shape: This is a pandas DataFrame attribute that returns a tuple representing the dimensions of the DataFrame in the format (number_of_rows, number_of_columns).

                    if outliers > 0:
                        self.logger.warning(f"{outliers} values below minimum for {column}")

                if max_val  is not None:
                    outliers = df[df[column]>max_val].shape[0] 

                    if outliers > 0:
                        self.logger.warning(f"{outliers} values above maximum for {column}")

        return df
    

    