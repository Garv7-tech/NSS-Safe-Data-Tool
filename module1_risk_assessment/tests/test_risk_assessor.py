import pytest
import pandas as pd
from module1_risk_assessment.risk_assessor import NSSRiskAssessor

# Note: The 'sample_dataframe' and 'test_config' fixtures are loaded from conftest.py

def test_detect_quasi_identifiers_happy_path(sample_dataframe, test_config):
    """Tests the 3-layer hybrid QI detection on normal data."""
    assessor = NSSRiskAssessor(test_config)
    detected_qis = assessor.detect_quasi_identifiers(sample_dataframe)
    expected_qis = ['Age', 'Sex', 'District_Code', 'Education_Level']
    for qi in expected_qis:
        assert qi in detected_qis
    assert 'Person_Serial_No' not in detected_qis
    assert 'Monthly_Consumer_Expenditure' not in detected_qis

def test_calculate_k_anonymity(sample_dataframe):
    """Tests the k-anonymity calculation."""
    assessor = NSSRiskAssessor({}) # Config not needed for this specific function
    qi_columns = ['Age', 'Sex', 'District_Code']
    # The group (Age=28, Sex=1, District_Code=101) is unique, so k=1
    assert assessor.calculate_k_anonymity(sample_dataframe, qi_columns) == 1

def test_calculate_k_anonymity_with_higher_k(sample_dataframe):
    """Tests k-anonymity where the smallest group is larger than 1."""
    assessor = NSSRiskAssessor({})
    # The group (Age=60, Sex=1, District_Code=104) appears twice
    # The group (Age=45, Sex=2, District_Code=103) appears twice
    # All other combinations with these QIs are unique, so k is still 1.
    # Let's test on a more limited set of QIs.
    qi_columns = ['Sex']
    # Count of Sex=1 is 6, count of Sex=2 is 5. So k=5.
    assert assessor.calculate_k_anonymity(sample_dataframe, qi_columns) == 5

def test_k_anonymity_on_empty_dataframe():
    """Tests that k-anonymity on an empty dataframe is 0."""
    assessor = NSSRiskAssessor({})
    empty_df = pd.DataFrame({'Age': []})
    assert assessor.calculate_k_anonymity(empty_df, ['Age']) == 0

def test_simulate_linkage_attack(sample_dataframe, test_config):
    """Tests that the linkage attack simulation runs and returns a valid report."""
    assessor = NSSRiskAssessor(test_config)
    qi_columns = ['Age', 'Sex', 'District_Code']
    report = assessor.simulate_linkage_attack(sample_dataframe, qi_columns)
    assert 'vulnerable_records_k1' in report
    assert report['vulnerable_records_k1'] > 0
    assert 'reidentification_rate_percent' in report