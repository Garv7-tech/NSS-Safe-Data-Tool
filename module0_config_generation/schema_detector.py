import pandas as pd
import logging
import re
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

# Heuristics to guess the *meaning* of a column
# This can be expanded over time.
SEMANTIC_HEURISTICS = {
    # Identifiers
    'hhid': (r'hhid|h_id|household_id|sample_household_number', 'identifier'),
    'person_id': (r'pid|p_id|person_serial_no|person_no|member_id', 'identifier'),
    
    # Common QIs
    'age': (r'age|age_yrs|h_age', 'qi'),
    'sex': (r'sex|gender|h_sex', 'qi'),
    'district': (r'dist|district|district_code|dist_code', 'qi'),
    'state': (r'state|state_ut_code|state_code', 'qi'),
    'sector': (r'sector|rural_urban|ru_sector', 'qi'),
    'education': (r'edu|education|general_education_level', 'qi'),
    'marital_status': (r'marital|m_status|marital_status', 'qi'),
    'social_group': (r'social_group|caste|soc_grp', 'qi'),
    'religion': (r'religion', 'qi'),

    # Sensitive Attributes
    'expenditure': (r'exp|expenditure|consump|mpce|monthly_consumer_expenditure', 'sensitive'),
    'income': (r'income|total_income|inc', 'sensitive'),
    'status_code': (r'status_code|principal_status|activity_status', 'qi'), # Also a QI
    'industry_code': (r'industry_code|nic_code|ind_code', 'qi'), # Also a QI
}

class SchemaDetector:
    """
    Analyzes an unknown DataFrame to infer its schema, data types,
    and semantic meaning (e.g., which column is 'Age').
    """
    def __init__(self, df: pd.DataFrame, survey_key: str):
        self.df = df
        self.survey_key = survey_key
        self.logger = logging.getLogger(f"{__name__}.{survey_key}")

    def _infer_dtype(self, col_name: str) -> str:
        """Infers a simplified data type."""
        dtype_str = str(self.df[col_name].dtype)
        if 'int' in dtype_str:
            return 'int'
        if 'float' in dtype_str:
            return 'float'
        if 'datetime' in dtype_str:
            return 'datetime'
        if 'category' in dtype_str:
            return 'category'
        return 'object' # Default to object/string

    def _infer_semantic_type(self, col_name: str) -> (str, str):
        """Guesses the semantic meaning and category of a column."""
        lowered_col = col_name.lower().strip()
        
        for semantic_type, (pattern, category) in SEMANTIC_HEURISTICS.items():
            if re.search(pattern, lowered_col):
                return semantic_type, category
        
        return 'unknown', 'unknown'

    def generate_survey_config(self) -> Dict[str, Any]:
        """Runs the full schema inference process."""
        self.logger.info(f"Generating new config for survey: '{self.survey_key}'")
        
        file_type_config = {
            'file_patterns': [f"*{self.survey_key}*.csv"], # A guess
            'required_columns': [],
            'identifier_columns': [],
            'dtypes': {},
            'date_columns': [],
            'missing_value_rules': {},
            'range_validation': {},
            'categorical_mappings': {},
            'column_mapping': {}
        }
        
        # For the master config
        qi_candidates = []
        analysis_identifiers = []
        analysis_demographics = []

        for col_name in self.df.columns:
            dtype = self._infer_dtype(col_name)
            semantic_type, category = self._infer_semantic_type(col_name)

            file_type_config['dtypes'][col_name] = dtype
            if dtype == 'datetime':
                file_type_config['date_columns'].append(col_name)
            
            # Map the auto-detected name to the standard name
            if semantic_type != 'unknown':
                file_type_config['column_mapping'][col_name] = semantic_type.title()

            if category == 'identifier':
                file_type_config['identifier_columns'].append(col_name)
                analysis_identifiers.append(col_name)
            
            if category == 'qi':
                qi_candidates.append(col_name)
                if semantic_type in ['age', 'sex']: # Age/Sex are always required
                    file_type_config['required_columns'].append(col_name)
                if semantic_type in ['age', 'sex', 'education', 'marital_status', 'social_group']:
                    analysis_demographics.append(col_name)

        # Build the final config blocks
        survey_config = {
            'survey_name': f"Auto-Generated {self.survey_key.upper()} Survey",
            'file_types': {
                'primary_file': file_type_config # Assume it's one big file
            }
        }
        
        detection_rules = {
            'file_patterns': {self.survey_key: [f"*{self.survey_key}*"]},
            'column_signatures': {self.survey_key: qi_candidates[:3]} # Use first 3 QIs
        }
        
        master_additions = {
            'risk_assessment': {'survey_specific': {self.survey_key: qi_candidates}},
            'analysis_columns': {
                'core_identifiers': {self.survey_key: analysis_identifiers},
                'demographics': {self.survey_key: analysis_demographics},
            }
        }

        return survey_config, detection_rules, master_additions