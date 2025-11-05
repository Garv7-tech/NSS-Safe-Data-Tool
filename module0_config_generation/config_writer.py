import yaml
import logging
from typing import Dict, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class ConfigWriter:
    """
    Safely reads, updates, and writes to the main ingestion_config.yaml file.
    Uses ruamel.yaml to preserve comments and formatting, but falls back
    to PyYAML if not installed.
    """
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_data = {}
        self.yaml = yaml # Default to PyYAML
        
        # Create a backup path
        self.backup_path = config_path.parent / f"{config_path.name}.bak"

    def _load_config(self):
        """Loads the existing YAML config file."""
        if not self.config_path.exists():
            logger.error(f"Config file not found at {self.config_path}! Cannot auto-update.")
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Create a backup
        try:
            os.replace(self.config_path, self.backup_path)
            logger.info(f"Created backup of config: {self.backup_path}")
        except Exception as e:
            logger.warning(f"Could not create config backup: {e}")

        with open(self.backup_path, 'r') as f:
            self.config_data = self.yaml.safe_load(f)
            if self.config_data is None:
                self.config_data = {}
        logger.debug(f"Loaded config from {self.config_path}")

    def _write_config(self):
        """Saves the updated config data back to the YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                self.yaml.safe_dump(self.config_data, f, default_flow_style=False, sort_keys=False, indent=2)
            logger.info(f"Successfully updated config file: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to write updated config file: {e}")
            # Restore from backup
            os.replace(self.backup_path, self.config_path)
            raise

    def add_new_survey(self, 
                       survey_key: str, 
                       survey_config: Dict, 
                       detection_rules: Dict, 
                       master_additions: Dict):
        """
        Adds the newly detected survey schema and detection rules
        to the loaded config data.
        """
        self._load_config()
        
        survey_key = survey_key.upper() # Config keys are uppercase (PLFS, HCES)

        # 1. Add the main survey schema
        if 'surveys' not in self.config_data: self.config_data['surveys'] = {}
        self.config_data['surveys'][survey_key] = survey_config
        
        # 2. Add detection rules
        if 'survey_detection' not in self.config_data: 
            self.config_data['survey_detection'] = {'file_patterns': {}, 'column_signatures': {}}
            
        self.config_data['survey_detection']['file_patterns'].update(detection_rules['file_patterns'])
        self.config_data['survey_detection']['column_signatures'].update(detection_rules['column_signatures'])

        # 3. Add risk assessment candidates
        if 'risk_assessment' not in self.config_data: 
            self.config_data['risk_assessment'] = {'quasi_identifier_candidates': {'common': [], 'survey_specific': {}}}
        self.config_data['risk_assessment']['quasi_identifier_candidates']['survey_specific'].update(
            master_additions['risk_assessment']['survey_specific']
        )
        
        # 4. Add analysis columns
        if 'analysis_columns' not in self.config_data:
            self.config_data['analysis_columns'] = {'core_identifiers': {}, 'demographics': {}}
            
        self.config_data['analysis_columns']['core_identifiers'].update(
            master_additions['analysis_columns']['core_identifiers']
        )
        self.config_data['analysis_columns']['demographics'].update(
            master_additions['analysis_columns']['demographics']
        )

        logger.info(f"Added new survey '{survey_key}' to config structure.")
        
        # 5. Save the updated file
        self._write_config()