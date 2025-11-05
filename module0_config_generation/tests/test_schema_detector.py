import pytest
import pandas as pd
from module0_config_generation.schema_detector import SchemaDetector

@pytest.fixture
def new_survey_df():
    """A sample DataFrame from a new, unknown survey."""
    data = {
        'hid': [101, 101, 102, 103],
        'member_id': [1, 2, 1, 1],
        'age_yrs': [45, 20, 33, 50],
        'gender': [1, 2, 1, 2],
        'dist_code': [10, 10, 11, 10],
        'total_income': [50000, 20000, 75000, 50000],
        'random_col': ['a', 'b', 'c', 'd']
    }
    return pd.DataFrame(data)

def test_schema_detector(new_survey_df):
    detector = SchemaDetector(new_survey_df, "new_education_survey")
    survey_config, detection_rules, master_additions = detector.generate_survey_config()

    # Test main survey config
    assert survey_config['survey_name'] == "Auto-Generated NEW_EDUCATION_SURVEY Survey"
    assert survey_config['file_types']['primary_file']['column_mapping']['age_yrs'] == 'Age'
    assert survey_config['file_types']['primary_file']['column_mapping']['dist_code'] == 'District'
    assert survey_config['file_types']['primary_file']['identifier_columns'] == ['hid', 'member_id']

    # Test detection rules
    assert detection_rules['file_patterns']['NEW_EDUCATION_SURVEY'] == ['*new_education_survey*']
    assert 'age_yrs' in detection_rules['column_signatures']['NEW_EDUCATION_SURVEY']
    
    # Test master additions
    assert 'age_yrs' in master_additions['risk_assessment']['survey_specific']['NEW_EDUCATION_SURVEY']
    assert 'gender' in master_additions['risk_assessment']['survey_specific']['NEW_EDUCATION_SURVEY']
    assert 'dist_code' in master_additions['risk_assessment']['survey_specific']['NEW_EDUCATION_SURVEY']
    assert 'total_income' not in master_additions['risk_assessment']['survey_specific']['NEW_EDUCATION_SURVEY'] # It's sensitive, not QI

    assert master_additions['analysis_columns']['core_identifiers']['NEW_EDUCATION_SURVEY'] == ['hid', 'member_id']
    assert master_additions['analysis_columns']['demographics']['NEW_EDUCATION_SURVEY'] == ['age_yrs', 'gender']