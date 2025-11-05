import pytest
import pandas as pd
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor
from conftest import sample_dataframe, test_config  # Assumes conftest.py is in root or accessible


# --- Fixture for Anonymized Data ---
@pytest.fixture(scope="session")
def anonymized_dataframe(sample_dataframe):
    """
    Provides a sample "anonymized" dataframe for testing.
    We will manually generalize Age and suppress some Education_Level.
    """
    anon_df = sample_dataframe.copy()

    # Manually generalize 'Age'
    # Bins: (0, 30], (30, 60]
    anon_df['Age'] = pd.cut(
        anon_df['Age'],
        bins=[0, 30, 60],
        right=True,
        labels=['1-30', '31-60']
    )

    # Manually suppress 'Education_Level' for some
    anon_df.loc[anon_df['District_Code'] == 101, 'Education_Level'] = '*'

    return anon_df


# --- Fixture for Utility Config ---
@pytest.fixture(scope="session")
def utility_config(test_config):
    """
    Adds the 'utility_assessment' block to the test config.
    """
    config = test_config.copy()
    config['utility_assessment'] = {
        'numeric_columns_to_compare': ['Age', 'Monthly_Consumer_Expenditure'],
        'ml_model': {
            'task_type': 'regression',
            'target_column': 'Monthly_Consumer_Expenditure',
            'feature_columns': ['Age', 'Sex', 'District_Code']
        }
    }
    return config


# --- Tests ---
def test_assessor_init(utility_config):
    """Tests that the assessor initializes correctly."""
    assessor = NSSUtilityAssessor(utility_config)
    assert assessor.config is not None
    assert 'ml_model' in assessor.config


def test_preprocess_anonymized_column(anonymized_dataframe):
    """Tests the conversion of ranges and strings to numeric."""
    assessor = NSSUtilityAssessor({})

    # Test with the generalized 'Age' column
    age_col = anonymized_dataframe['Age']
    numeric_age = assessor._preprocess_anonymized_column(age_col)

    # '1-30' -> 15.5
    # '31-60' -> 45.5
    assert numeric_age.iloc[0] == 15.5  # Original was 25
    assert numeric_age.iloc[1] == 15.5  # Original was 30
    assert numeric_age.iloc[3] == 45.5  # Original was 45

    # Test with a '*' column
    edu_col = anonymized_dataframe['Education_Level']
    numeric_edu = assessor._preprocess_anonymized_column(edu_col)
    assert pd.isna(numeric_edu.iloc[0])  # Original was 'Grad', now '*' -> NaN
    assert pd.isna(numeric_edu.iloc[9])  # Original was 'Grad', now '*' -> NaN


def test_statistical_comparison(sample_dataframe, anonymized_dataframe, utility_config):
    """Tests the basic statistics comparison."""
    assessor = NSSUtilityAssessor(utility_config)
    report = assessor._compare_basic_statistics(sample_dataframe, anonymized_dataframe)

    assert 'Age' in report
    assert 'Monthly_Consumer_Expenditure' in report

    # Check Age (which was generalized)
    assert report['Age']['original_mean'] == pytest.approx(38.8, abs=0.1)
    assert report['Age']['anonymized_mean'] == pytest.approx(28.5, abs=0.1)
    assert report['Age']['mean_diff_percent'] < -25.0

    # Check Expenditure (which was NOT generalized)
    assert report['Monthly_Consumer_Expenditure']['original_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['anonymized_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['mean_diff_percent'] == pytest.approx(0.0)


def test_correlation_comparison(sample_dataframe, anonymized_dataframe, utility_config):
    """Tests the correlation matrix comparison."""
    assessor = NSSUtilityAssessor(utility_config)
    report = assessor._compare_correlation(sample_dataframe, anonymized_dataframe)

    assert 'mean_absolute_correlation_difference' in report
    # The correlation will change because Age was binned, breaking its
    # linear relationship with Expenditure. The diff should be > 0.
    assert report['mean_absolute_correlation_difference'] > 0.05


def test_ml_model_comparison(sample_dataframe, anonymized_dataframe, utility_config):
    """Tests the ML utility comparison."""
    assessor = NSSUtilityAssessor(utility_config)
    report = assessor._compare_ml_model_performance(sample_dataframe, anonymized_dataframe)

    assert report['task_type'] == 'regression'
    assert report['model'] == 'LinearRegression'
    assert 'original_score (r2_score)' in report
    assert 'anonymized_score (r2_score)' in report

    # Original score should be higher than the anonymized one
    assert report['original_score (r2_score)'] > report['anonymized_score (r2_score)']
    assert report['performance_drop_percent'] > 0.0


def test_final_linkage_attack(anonymized_dataframe, utility_config):
    """Tests the final linkage attack on the anonymized data."""
    assessor = NSSUtilityAssessor(utility_config)
    # Use the QIs that were not generalized for a fair test
    qis = ['Sex', 'District_Code']
    report = assessor._simulate_linkage_attack(anonymized_dataframe, qis)

    assert 'final_k_anonymity' in report
    assert 'vulnerable_records_k1' in report

    # For (Sex, District_Code), the smallest group is 1
    # (Sex=2, District_Code=101) is unique
    assert report['final_k_anonymity'] == 1
    assert report['vulnerable_records_k1'] > 0
