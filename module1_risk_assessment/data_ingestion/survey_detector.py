"""
NSS Survey Type Auto-Detection
Automatically detects survey type from uploaded files without changing existing code
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from .utils import setup_logging

class NSSSurveyDetector:
    """
    Automatically detects NSS survey type from file patterns and column signatures
    """

    def __init__(self, config : Dict):
        """
        Initialize detector with full multi-survey configuration
        
        Args:
            config: Complete configuration dictionary with all surveys
        """
        self.config = config
        self.logger = setup_logging(__name__)

    def detect_survey_type(self, input_dir: str)->str :
        """
        Auto-detect survey type from files in input-directory

        Args:
            input_dir : Directory containing NSS files

        Returns:
            Detected survery type (PLFS, HCES, ASI, EUS, etc.)
        """

        input_path = Path(input_dir)

        # Get all CSV files in directory
        csv_files = list(input_path.glob('*.csv'))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in input directory")
        
        self.logger.info(f"Found {len(csv_files)} CSV files for detection")

        # Method 1 : Check filename patterns
        survey_from_filenames = self._detect_from_filenames(csv_files)
        if survey_from_filenames:
            self.logger.info(f"Survey detected from filenames: {survey_from_filenames}")
            return survey_from_filenames
        
        # Method 2 : Check column signatures
        survey_from_columns = self._detect_from_columns(csv_files)
        if survey_from_columns:
            self.logger.info(f"Survey detected from columns: {survey_from_columns}")
            return survey_from_columns
        
        # Fallback to default 
        default_survey = self.config.get('default_survey', 'PLFS')
        self.logger.warning(f"Could not auto-detect survey type, using default: {default_survey}")
        return default_survey
    

    def _detect_from_filenames(self, csv_files : List[Path])->Optional[str]:
        """Detect survey types from filename patterns"""

        detection_rules = self.config.get('survey_detection',{}).get('file_patterns', {})

        for survey_type, patterns in detection_rules.items():
            for file_path in csv_files:
                filename = file_path.name.lower()
                
                for pattern in patterns:
                    if pattern.lower().replace('*', '') in filename:
                        return survey_type
        return None
    
    def _detect_from_columns(self, csv_files: List[Path])-> Optional[str]:
        """Detect survey type from column signatures"""

        detection_rules = self.config.get('survey_detection',{}).get('column_signatures',{})

        # Read first few rows of each file to check columns
        for file_path in csv_files:
            try:
                # Read just the header
                sample_df = pd.read_csv(file_path, nrows = 0)
                file_columns = set(sample_df.columns)

                # Check which survey signature matches best
                for survey_type, signature_columns in detection_rules.items():
                    signature_set = set(signature_columns)

                    # If most signature columns are present, it's likely this survey
                    if len(signature_set.intersection(file_columns)) >= len(signature_set) * 0.7: 
                        return survey_type

            except Exception as e:
                self.logger.warning(f"Could not read file {file_path} for column detection : {e}")
                continue

        return None


class NSSConfigResolver:
    """
    Resolves multi-survey config to format expectec by existing code
    """

    def __init__(self, config: Dict):
        """
        Initialize resolver with full multi-survey configuration


        Args :
            config : Complete configuration dictionary with all surveys
        """

        self.config = config
        self.logger = setup_logging(__name__)

    
    def resolve_config_for_survey(self, survey_type: str)-> Dict:
        """
        Resolve config for specific survey type to format expected by existing code

        Args : 
            survey_type : Survey type (PLFS, HCES, etc.)

        Returns :
            Resolved config dictionary in format expected by existing parser/cleaner/merger
        """

        if survey_type not in self.config['surveys']:
            raise ValueError(f"Survey type '{survey_type}' not found in configuration")
        
        survey_config = self.config['surveys'][survey_type]

        # Build resolved config that existing code expects
        resolved_config = {
            # Global settings (unchanged)
            'encoding' : self.config.get('encoding', 'utf-8'),
            'na_values' : self.config.get('na_values', []),
            'chunk_size' : self.config.get('chunk_size', 10000),
            'output_format' : self.config.get('output_format', 'parquet'),
            
            # Survey-specific file_types (what existing code reads)
            'file_types' : survey_config['file_types'],
            
            # Resolved merge_keys (what existing merger reads)
            'merge_keys' : self._resolve_merge_keys(survey_type),

            # Resolved analysis_columns (what existing merger reads)
            'analysis_columns' : self._resolve_analysis_columns(survey_type),

            # Pass through other settings
            'file_processing' : self.config.get('file_processing', {}),
            'quality_checks' : self.config.get('quality_checks', {}),
            'output' : self.config.get('output', {})
        }

        self.logger.info(f"Config resolved for survey type: {survey_type}")
        return resolved_config
    
    def _resolve_merge_keys(self, survey_type: str)-> Dict:
        """Resolve merge key for survey type"""

        merge_strategies = self.config.get('merge_strategies', {})

        resolved_merge_keys = {}

        for merge_type, strategy in merge_strategies.items():
            if survey_type in strategy.get('survey_specific', {}):
                resolved_merge_keys[merge_type] = strategy['survey_specific'][survey_type]

            else : 
                resolved_merge_keys[merge_type] = strategy['default_keys', []]

        return resolved_merge_keys


    def _resolve_analysis_columns(self, survey_type: str)-> Dict : 
        """Resolve analysis columns for survey type"""

        analysis_config = self.config.get('analysis_columns', {})

        resolved_analysis = {}

        for category, columns_config in analysis_config.items():
            if isinstance(columns_config, dict):
                if survey_type in columns_config:
                    resolved_analysis[category] = columns_config[survey_type]
                elif 'common' in columns_config:
                    resolved_analysis[category] = columns_config['common']

                else:
                    resolved_analysis[category] = []
            
            else:
                resolved_analysis[category] = columns_config

        return resolved_analysis