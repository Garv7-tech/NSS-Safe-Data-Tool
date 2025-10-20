import pytest
import pandas as pd
from module2_privacy_enhancement.privacy_enhancer import NSSPrivacyEnhancer
# Import fixtures from the conftest file located in the module1 tests directory
from conftest import sample_dataframe, test_config

def test_anonymization_achieves_target_k(sample_dataframe, test_config):
    """Tests if the main k-anonymity process achieves the target k."""
    risk_report = {'detected_quasi_identifiers': ['Age', 'Sex', 'District_Code']}
    enhancer = NSSPrivacyEnhancer(test_config, risk_report)
    target_k = test_config['privacy_enhancement']['goals']['k_anonymity']['target_k']
    
    anonymized_df = enhancer.anonymize(sample_dataframe.copy())

    assert not sample_dataframe.equals(anonymized_df), "Anonymized data should not be the same as original"
    
    final_k = enhancer._calculate_k_anonymity(anonymized_df)
    assert final_k >= target_k, f"Final k ({final_k}) did not meet target k ({target_k})"

def test_anonymization_fails_when_info_loss_is_too_high(sample_dataframe, test_config):
    """
    Tests that if goals cannot be met, the original dataframe is returned.
    We set max_info_loss to an impossibly low value.
    """
    # Set an impossible goal for info loss
    test_config['privacy_enhancement']['goals']['k_anonymity']['max_info_loss'] = 0.01
    
    risk_report = {'detected_quasi_identifiers': ['Age', 'Sex', 'District_Code']}
    enhancer = NSSPrivacyEnhancer(test_config, risk_report)

    anonymized_df = enhancer.anonymize(sample_dataframe.copy())

    # The enhancer should log an error and return the original dataframe
    assert sample_dataframe.equals(anonymized_df), "Original dataframe should be returned when goals are not met"

def test_normalized_certainty_penalty_calculation(sample_dataframe, test_config):
    """Tests the information loss (NCP) metric calculation."""
    risk_report = {'detected_quasi_identifiers': ['Age', 'District_Code']}
    enhancer = NSSPrivacyEnhancer(test_config, risk_report)
    
    anonymized_df = sample_dataframe.copy()
    anonymized_df['Age'] = '25-60' # Manually generalize all Age values
    
    info_loss = enhancer._calculate_normalized_certainty_penalty(sample_dataframe, anonymized_df)
    
    assert 0 < info_loss < 1, "Info loss should be between 0 and 1"
    # NCP for one fully generalized numeric column out of two QIs should be ~0.5
    assert pytest.approx(0.5, abs=0.1) == info_loss

def test_anonymize_with_synthetic_data(sample_dataframe, test_config):
    """Tests the synthetic data generation strategy."""
    # This test requires the 'sdv' library to be installed
    pytest.importorskip("sdv")
    
    test_config['privacy_enhancement']['privacy_strategy'] = 'synthetic_data'
    risk_report = {'detected_quasi_identifiers': list(sample_dataframe.columns)}
    enhancer = NSSPrivacyEnhancer(test_config, risk_report)
    
    synthetic_df = enhancer.anonymize(sample_dataframe)
    
    assert synthetic_df.shape == sample_dataframe.shape
    assert all(synthetic_df.columns == sample_dataframe.columns)
    assert not sample_dataframe.equals(synthetic_df)