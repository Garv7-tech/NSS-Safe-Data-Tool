import pytest
import pandas as pd

@pytest.fixture(scope="session")
def sample_dataframe():
    """
    Provides a standard, clean DataFrame for all tests across all modules.
    This fixture is created only ONCE per test session.
    """
    data = {
        'Age': [25, 30, 25, 45, 50, 45, 30, 60, 60, 25, 28],
        'Sex': [1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1],
        'District_Code': [101, 102, 101, 103, 102, 103, 102, 104, 104, 101, 101],
        'Education_Level': ['Grad', 'PG', 'Grad', 'HS', 'PG', 'HS', 'PG', 'Doc', 'Doc', 'Grad', 'Grad'],
        'Monthly_Consumer_Expenditure': [5000, 8000, 5200, 12000, 15000, 11000, 8500, 20000, 18000, 4800, 6000],
        'Person_Serial_No': [1, 1, 2, 1, 1, 2, 2, 1, 2, 3, 4] # Direct Identifier
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def test_config():
    """
    Provides a standard configuration dictionary for all tests.
    This fixture is created only ONCE per test session.
    """
    config = {
        'survey_type_detected': 'PLFS',
        
        # --- Module 1 Config ---
        'risk_assessment': {
            'quasi_identifier_candidates': {
                'common': ['Age', 'Sex', 'District_Code'],
                'survey_specific': {'PLFS': ['Education_Level']}
            }
        },
        
        # --- Module 2 Config ---
        'privacy_enhancement': {
            'privacy_strategy': 'k_anonymity',
            'goals': {
                'k_anonymity': {
                    'target_k': 3,
                    'max_info_loss': 0.5
                }
            }
        }
        
        # --- Module 3 Config (Future) ---
        # 'utility_assessment': {
        #     'metrics': ['cosine_similarity', 'kl_divergence']
        # }
    }
    return config