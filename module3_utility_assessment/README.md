# Module 3: Utility Measurement

## 1. Overview

This module is the final diagnostic step in the NSS SafeData Pipeline. Its sole purpose is to measure the analytical utility of the anonymized data produced by Module 2. It answers the question: "How much analytical value did we lose in exchange for privacy?"

It works by taking both the original data and the anonymized data as inputs and running a series of comparisons between them, fulfilling the requirements of the `PS-1.pdf` problem statement.

***

## 2. Core Components

The logic for this module is encapsulated within the `NSSUtilityAssessor` class, located in `utility_assessor.py`.

### `NSSUtilityAssessor`

This class is the engine for all utility measurement tasks. Its operations are driven by the `utility_assessment` section in the `ingestion_config.yaml` file.

#### Key Responsibilities

**A. Statistical Comparison (`_compare_basic_statistics`)**  
* **What it does:** Compares basic descriptive statistics (mean, standard deviation) for key numerical columns, as required by `PS-1.pdf`.  
* **How it works:** It reads the `numeric_columns_to_compare` list from the config. For each column, it calculates the stats for both datasets and reports the percentage difference.  
* **Handling Generalization:** It's smart enough to handle anonymized values. A range like `'21-25'` (from k-anonymity) is converted to its midpoint `23.0` for calculation using `_preprocess_anonymized_column`.

**B. Distribution Comparison (New - Fulfills PS-1.pdf)**  
* **What it does:** Fulfills the `PS-1.pdf` requirement to compare distributions.  
* **How it works:** It uses the Kolmogorov-Smirnov (KS) two-sample test (from `scipy.stats`) on key numeric columns.  
* **Metric:** This test yields a p-value. A high p-value (e.g., `> 0.05`) suggests the anonymized data's distribution is statistically similar to the original. A low p-value suggests the distribution has changed significantly.

**C. Correlation Comparison (`_compare_correlation`)**  
* **What it does:** Measures if the relationships between variables were preserved.  
* **How it works:** It calculates the Pearson correlation matrix for both datasets and reports the Mean Absolute Difference. A low score is better.

**D. Machine Learning (ML) Utility (`_compare_ml_model_performance`)**  
* **What it does:** This is the most practical test. It checks if the anonymized data can still be used for a real-world predictive modeling task.  
* **How it works:**  
  1. The user defines an ML task in the config (e.g., "predict 'Expenditure' using 'Age' and 'Sex'").  
  2. The assessor trains a simple model (`LinearRegression`) on the original data to get a "baseline" R² score (e.g., `0.90`).  
  3. It then trains the same model on the anonymized data to get a new score (e.g., `0.85`).  
  4. It reports the final performance drop (e.g., `5.55%`).

**E. Final Risk Assessment (`_simulate_linkage_attack`)**  
* **What it does:** This function re-runs the k-anonymity check from Module 1, but this time on the final anonymized data.  
* **Goal:** This is a "pass/fail" check on Module 2. It reports the final `k-anonymity` score and the number of remaining unique records (`k=1`). Ideally, after anonymization, this number should be `0`.

***

## 3. Configuration

This module is controlled by the `utility_assessment` block in the `ingestion_config.yaml`.

```yaml
utility_assessment:
  # Columns for stats & distribution checks
  numeric_columns_to_compare:
    - 'Age'
    - 'Household_Size'
    - 'Monthly_Consumer_Expenditure'

  # Config for the ML utility test
  ml_model:
    task_type: 'regression'
    target_column: 'Monthly_Consumer_Expenditure'
    feature_columns:
      - 'Age'
      - 'Sex'
      - 'Social_Group'
      - 'Household_Size'
```


***

## 4. How to Use

The `NSSUtilityAssessor` is called from `main.py` after Module 2 has finished.

### Inputs

- `original_df`: The clean, original Pandas DataFrame.
- `anonymized_df`: The anonymized Pandas DataFrame from Module 2.
- `config`: The resolved configuration dictionary.
- `qi_columns`: The list of QIs from Module 1's report.


### Output

A comprehensive JSON "Utility Report" containing the results of all tests.

***

## 5. Module 3 Test: `module3_utility_assessment/tests/test_utility_assessor.py`

This is the complete test file that replaces your old one. It tests all features of the new `utility_assessor.py`, including the distribution check.

**File:** `module3_utility_assessment/tests/test_utility_assessor.py`

```python
import pytest
import pandas as pd
import numpy as np
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor
from conftest import sample_dataframe, full_master_config_dict  # Import fixtures


@pytest.fixture
def resolved_plfs_config(full_master_config_dict):
    """
    A fixture to simulate the 'resolved_config' for PLFS,
    which includes the 'utility_assessment' block.
    """
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
    anon_df['Education_Level'] = '*'  # Suppress all for simplicity

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
    assessor = NSSUtilityAssessor({})  # No config needed for this helper

    age_col = anonymized_dataframe['Age']
    numeric_age = assessor._preprocess_anonymized_column(age_col)
    assert numeric_age.iloc[0] == 25.5  # '21-30'
    assert numeric_age.iloc[3] == 45.5  # '41-50'

    edu_col = anonymized_dataframe['Education_Level']
    numeric_edu = assessor._preprocess_anonymized_column(edu_col)
    assert pd.isna(numeric_edu.iloc[0])  # '*' -> NaN

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

    assert report['Age']['original_mean'] == pytest.approx(38.8, abs=0.1)
    assert report['Age']['anonymized_mean'] == pytest.approx(32.27, abs=0.1)
    assert report['Age']['mean_diff_percent'] < -16.0

    assert report['Monthly_Consumer_Expenditure']['original_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['anonymized_mean'] == pytest.approx(9800, abs=1)
    assert report['Monthly_Consumer_Expenditure']['mean_diff_percent'] == pytest.approx(0.0)


def test_distribution_comparison(sample_dataframe, anonymized_dataframe, resolved_plfs_config):
    """Tests the KS test for distribution comparison."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    report = assessor._compare_distributions(sample_dataframe, anonymized_dataframe)

    assert 'Age' in report
    assert 'Monthly_Consumer_Expenditure' in report

    assert report['Age']['p_value'] < 0.05
    assert report['Age']['are_distributions_statistically_similar'] is False

    assert report['Monthly_Consumer_Expenditure']['p_value'] == 1.0
    assert report['Monthly_Consumer_Expenditure']['are_distributions_statistically_similar'] is True


def test_ml_model_comparison(sample_dataframe, anonymized_dataframe, resolved_plfs_config):
    """Tests the ML utility comparison."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)
    report = assessor._compare_ml_model_performance(sample_dataframe, anonymized_dataframe)

    assert report['status'] == 'success'
    assert report['task_type'] == 'regression'
    assert report['model'] == 'LinearRegression'
    assert 'original_score (r2_score)' in report
    assert 'anonymized_score (r2_score)' in report

    assert report['original_score (r2_score)'] >= report['anonymized_score (r2_score)']
    assert report['performance_drop_percent'] >= 0.0


def test_final_risk_check(anonymized_dataframe, resolved_plfs_config):
    """Tests the final k-anonymity check on the anonymized data."""
    assessor = NSSUtilityAssessor(resolved_plfs_config)

    qis = resolved_plfs_config['risk_assessment']['quasi_identifier_candidates']['common'] + \
          resolved_plfs_config['risk_assessment']['quasi_identifier_candidates']['survey_specific']['PLFS']

    report = assessor._check_final_risk(anonymized_dataframe, qis)

    assert report['final_k_anonymity'] == 1
    assert report['vulnerable_records_k1'] == 2
```
