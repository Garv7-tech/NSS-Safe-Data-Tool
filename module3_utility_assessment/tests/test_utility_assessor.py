import pytest
import pandas as pd
import numpy as np
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor
from conftest import sample_dataframe, full_master_config_dict # Import fixtures

@pytest.fixture
def resolved_plfs_config(full_master_config_dict):
    """
    A fixture to simulate the 'resolved_config' for PLFS,
    which includes the 'utility_assessment' block.
    """
    # This simulates the output of NSSConfigResolver
    config = {
        'survey_type_detected': 'PLFS',
        'utility_assessment': full_master_config_dict['utility_assessment'],
        'analysis_columns': full_master_config_dict['analysis_columns']
        # Add other keys if needed by the class under test
    }
    return config

@pytest.fixture(scope="module")
def anonymized_dataframe(sample_dataframe):
    """
    Provides a sample "anonymized" dataframe for testing.
    We will manually generalize Age and suppress some Education_Level.
    This is scoped to 'module' to avoid recreating it for every test.
    """
    anon_df = sample_dataframe.copy()

    # Manually generalize 'Age' into ranges
    # Original: [25, 30, 25, 45, 50, 45, 30, 60, 60, 25, 28]
    # Becomes:
    anon_df['Age'] = [
        '21-30', '21-30', '21-30', '41-50', '41-50', '41-50', '21-30', 
        '51-60', '51-60', '21-30', '21-30'
    ]

    # Manually suppress 'Education_Level'
    anon_df['Education_Level'] = '*' # Suppress all for simplicity
    
    # Manually generalize 'Household_Size'
    anon_df['Household_Size'] = '2-4'

    return anon_df


def test_assessor_init(resolved_plfs_config):
    """Tests that the assessor initializes correctly."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    assert assessor.config is not None
    assert 'ml_model' in assessor.config
    assert assessor.numeric_cols == ['Age', 'Monthly_Consumer_Expenditure', 'Household_Size']

def test_preprocess_anonymized_column(anonymized_dataframe):
    """Tests the conversion of ranges and strings to numeric."""
    assessor = NSSUtilityAssessor({}) # No config needed for this helper

    # Test with the generalized 'Age' column: '21-30' -> 25.5
    age_col = anonymized_dataframe['Age']
    numeric_age = assessor._preprocess_anonymized_column(age_col)
    assert numeric_age.iloc[0] == 25.5 # '21-30'
    assert numeric_age.iloc[3] == 45.5 # '41-50'

    # Test with a '*' column
    edu_col = anonymized_dataframe['Education_Level']
    numeric_edu = assessor._preprocess_anonymized_column(edu_col)
    assert pd.isna(numeric_edu.iloc[0])  # '*' -> NaN

    # Test with a simple string number
    series_str = pd.Series(['42', '55.5'])
    numeric_str = assessor._preprocess_anonymized_column(series_str)
    assert numeric_str.iloc[0] == 42.0
    assert numeric_str.iloc[1] == 55.5

def test_statistical_comparison(sample_dataframe, anonymized_dataframe, resolved_plfs_config):
    """Tests the basic statistics comparison."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    report = assessor._compare_basic_statistics(sample_dataframe, anonymized_dataframe)

    assert 'Age' in report
    assert 'Monthly_Consumer_Expenditure' in report

    # Check Age (which was generalized)
    assert report['Age']['original_mean'] == pytest.approx(38.8, abs=0.1)
    # Anonymized mean: (25.5 * 6 + 45.5 * 2 + 55.5 * 2) / 11 = (153 + 91 + 111) / 11 = 355 / 11 = 32.27
    assert report['Age']['anonymized_mean'] == pytest.approx(32.27, abs=0.1)
    assert report['Age']['mean_diff_percent'] < -16.0

    # Check Expenditure (which was NOT generalized in anon_df)
    assert report['Monthly_Consumer_Expenditure']['original_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['anonymized_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['mean_diff_percent'] == pytest.approx(0.0)

def test_distribution_comparison(sample_dataframe, anonymized_dataframe, resolved_plfs_config):
    """Tests the KS test for distribution comparison."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    report = assessor._compare_distributions(sample_dataframe, anonymized_dataframe)

    assert 'Age' in report
    assert 'Monthly_Consumer_Expenditure' in report

    # Age distribution was changed, so p-value should be low (<0.05)
    # and they should be NOT similar.
    assert report['Age']['p_value'] < 0.05
    assert report['Age']['are_distributions_statistically_similar'] == False

    # Expenditure was unchanged, so p-value should be high (1.0)
    # and they SHOULD be similar.
    assert report['Monthly_Consumer_Expenditure']['p_value'] == 1.0
    assert report['Monthly_Consumer_Expenditure']['are_distributions_statistically_similar'] == True

def test_ml_model_comparison(sample_dataframe, anonymized_dataframe, resolved_plfs_config):
    """Tests the ML utility comparison."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    report = assessor._compare_ml_model_performance(sample_dataframe, anonymized_dataframe)

    assert report['status'] == 'success'
    assert report['task_type'] == 'regression'
    assert report['model'] == 'LinearRegression'
    assert 'original_score (r2_score)' in report
    assert 'anonymized_score (r2_score)' in report

    # Original score should be higher (or equal) to the anonymized one
    assert report['original_score (r2_score)'] >= report['anonymized_score (r2_score)']
    assert report['performance_drop_percent'] >= 0.0

def test_final_risk_check(anonymized_dataframe, resolved_plfs_config):
    """Tests the final k-anonymity check on the anonymized data."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    
    # QIs are defined by the config loaded into the assessor
    qis = resolved_plfs_config['risk_assessment']['quasi_identifier_candidates']['common'] + \
        resolved_plfs_config['risk_assessment']['quasi_identifier_candidates']['survey_specific']['PLFS']
    
    # QIs: ['Age', 'Sex', 'District_Code', 'Education_Level', 'Social_Group']
    report = assessor._check_final_risk(anonymized_dataframe, qis)

    # Groups in anonymized_dataframe:
    # 'Age' is generalized (e.g., '21-30')
    # 'Education_Level' is generalized to '*'
    # 'Household_Size' is generalized to '2-4'
    #
    # QIs to group by: ['Age', 'Sex', 'District_Code', 'Education_Level', 'Social_Group']
    #
    # 1. ('21-30', 1, 101, '*', 1) - Rows [0, 2, 10] -> count 3
    # 2. ('21-30', 2, 102, '*', 2) - Rows [1, 6] -> count 2
    # 3. ('41-50', 2, 103, '*', 3) - Rows [3, 5] -> count 2
    # 4. ('41-50', 1, 102, '*', 2) - Row [4] -> count 1  <- VULNERABLE
    # 5. ('51-60', 1, 104, '*', 9) - Rows [7, 8] -> count 2
    # 6. ('21-30', 2, 101, '*', 1) - Row [9] -> count 1  <- VULNERABLE
    
    # Smallest group is k=1
    assert report['final_k_anonymity'] == 1
    assert report['vulnerable_records_k1'] == 2