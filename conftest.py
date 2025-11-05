import pytest
import pandas as pd
from pydantic import ValidationError
from module4_reporting.config_models import MasterConfig # Import the validator

@pytest.fixture(scope="session")
def sample_dataframe():
    """
    Provides a standard, clean DataFrame for all tests across all modules.
    This fixture is created only ONCE per test session.
    """
    data = {
        'Panel': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Sample_Household_Number': [101, 102, 101, 103, 102, 103, 102, 104, 104, 101, 101],
        'Person_Serial_No': [1, 1, 2, 1, 1, 2, 2, 1, 2, 3, 4], # Direct Identifier
        'Age': [25, 30, 25, 45, 50, 45, 30, 60, 60, 25, 28],
        'Sex': [1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1], # 1:M, 2:F
        'District_Code': [101, 102, 101, 103, 102, 103, 102, 104, 104, 101, 101],
        'Education_Level': ['Grad', 'PG', 'Grad', 'HS', 'PG', 'HS', 'PG', 'Doc', 'Doc', 'Grad', 'Grad'],
        'Social_Group': [1, 2, 1, 3, 2, 3, 2, 9, 9, 1, 1],
        'Monthly_Consumer_Expenditure': [5000, 8000, 5200, 12000, 15000, 11000, 8500, 20000, 18000, 4800, 6000],
        'Household_Size': [4, 2, 4, 3, 2, 3, 2, 2, 2, 4, 4]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def full_master_config_dict():
    """
    Provides a full, valid MasterConfig dictionary that mimics
    the 'ingestion_config.yaml' file. This is crucial for all
    modules to pull their configs from.
    """
    config_dict = {
        'encoding': 'utf-8',
        'na_values': ['', 'NA', 'NULL'],
        'chunk_size': 10000,
        'output_format': 'parquet',
        'default_survey': 'PLFS',
        
        'surveys': {
            'PLFS': {
                'survey_name': "Periodic Labour Force Survey",
                'file_types': {
                    'household': {
                        'file_patterns': ['*hh*.csv'],
                        'required_columns': ['Sample_Household_Number', 'District_Code'],
                        'identifier_columns': ['Sample_Household_Number'],
                        'dtypes': {'District_Code': 'int16', 'Monthly_Consumer_Expenditure': 'float32'},
                        'date_columns': [],
                        'missing_value_rules': {'Monthly_Consumer_Expenditure': 'fill_zero'},
                        'range_validation': {'Household_Size': {'min': 1, 'max': 20}},
                        'categorical_mappings': {'Sector': {1: 'Rural', 2: 'Urban'}}
                    },
                    'person': {
                        'file_patterns': ['*per*.csv'],
                        'required_columns': ['Person_Serial_No', 'Age', 'Sex'],
                        'identifier_columns': ['Person_Serial_No'],
                        'dtypes': {'Age': 'int8', 'Sex': 'int8'},
                        'missing_value_rules': {'Age': 'drop_rows'},
                        'range_validation': {'Age': {'min': 0, 'max': 120}},
                        'categorical_mappings': {'Sex': {1: 'Male', 2: 'Female'}}
                    }
                }
            }
        },
        
        'merge_strategies': {
            'household_person': {
                'default_keys': ['HHID'],
                'survey_specific': {
                    'PLFS': ['Panel', 'Sample_Household_Number', 'State_Ut_Code', 'District_Code']
                }
            }
        },
        
        'analysis_columns': {
            'core_identifiers': {
                'PLFS': ['Panel', 'Sample_Household_Number', 'Person_Serial_No']
            },
            'demographics': {
                'common': ['Age', 'Sex'],
                'survey_specific': {
                    'PLFS': ['Marital_Status', 'Social_Group', 'Education_Level']
                }
            },
            'household_characteristics': {
                'common': ['Household_Size', 'Sector'],
                'survey_specific': {
                    'PLFS': ['Monthly_Consumer_Expenditure']
                }
            },
            'employment': {'PLFS': ['Principal_Status_Code']},
            'expenditure': {}
        },
        
        'survey_detection': {
            'file_patterns': {'PLFS': ['*per*.csv', '*hh*.csv']},
            'column_signatures': {'PLFS': ['Panel', 'Sample_Household_Number']}
        },
        
        'file_processing': {
            'max_file_size_mb': 500,
            'supported_formats': ['csv'],
            'auto_detect_delimiter': True,
            'delimiter_candidates': [',']
        },
        
        'quality_checks': {
            'duplicate_detection': {'enabled': True},
            'missing_data_threshold': 0.9,
            'outlier_detection': {'enabled': True},
            'referential_integrity': {'enabled': True}
        },
        
        'output': {
            'save_intermediate': False,
            'intermediate_formats': ['parquet'],
            'final_formats': ['parquet', 'csv'],
            'include_metadata': True,
            'create_data_dictionary': True
        },
        
        'risk_assessment': {
            'quasi_identifier_candidates': {
                'common': ['Age', 'Sex', 'District_Code'],
                'survey_specific': {
                    'PLFS': ['Education_Level', 'Social_Group']
                }
            }
        },
        
        'privacy_enhancement': {
            'privacy_strategy': 'k_anonymity',
            'goals': {
                'k_anonymity': {'target_k': 3, 'max_info_loss': 0.5},
                'differential_privacy': {'epsilon': 1.0}
            }
        },
        
        'utility_assessment': {
            'numeric_columns_to_compare': ['Age', 'Monthly_Consumer_Expenditure', 'Household_Size'],
            'ml_model': {
                'task_type': 'regression',
                'target_column': 'Monthly_Consumer_Expenditure',
                'feature_columns': ['Age', 'Sex', 'District_Code', 'Education_Level', 'Household_Size']
            }
        }
    }
    
    # Validate the config dict against the Pydantic model
    try:
        MasterConfig(**config_dict)
    except ValidationError as e:
        print("FATAL ERROR in conftest.py: The mock config dictionary is invalid!")
        print(e)
        raise e
        
    return config_dict