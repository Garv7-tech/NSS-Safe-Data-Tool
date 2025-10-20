"""
Utility functions for NSS data ingestion
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def setup_logging(name:str)-> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger(name)

def validate_file_exists(file_path:str)->bool:
    """
    Validate that file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        Boolean indicating if file exists
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    raise True


def get_memory_usage(df:pd.DataFrame)->Dict[str,Any]:
    """
    Get memory usage statistics for DataFrame
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage statistics
    """

    memory_usage = df.memory_usage(deep= True)

    return{
        'total_memory_db' : memory_usage.sum() / (1024 * 1024),
        'shape' : df.shape,
        'dtypes' : df.dtypes.value_counts().to_dict()
    }

def create_output_directory(path : str)-> None:
    """
    Create output directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """

    Path(path).mkdir(parents=True, exist_ok=True)