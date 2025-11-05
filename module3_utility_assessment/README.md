# Module 3: Utility Measurement

## 1. Overview

This module is the final diagnostic step in the NSS SafeData Pipeline. Its sole purpose is to measure the **analytical utility** of the anonymized data produced by Module 2. It answers the question: "How much analytical value did we lose in exchange for privacy?"

It works by taking both the **original data** and the **anonymized data** as inputs and running a series of comparisons between them.

***

## 2. Core Components

The logic for this module is encapsulated within the `NSSUtilityAssessor` class, located in `utility_assessor.py`.

### `NSSUtilityAssessor`

This class is the engine for all utility measurement tasks. Its operations are driven by the `utility_assessment` section in the `ingestion_config.yaml` file.

#### Key Responsibilities:

**A. Statistical Comparison (`_compare_basic_statistics`)**
* **What it does:** Compares basic descriptive statistics (mean, median, standard deviation) for key numerical columns.
* **How it works:** It reads the `numeric_columns_to_compare` list from the config. For each column, it calculates the stats for both datasets and reports the percentage difference.
* **Handling Generalization:** It's smart enough to handle anonymized values. A range like `'21-25'` is converted to its midpoint `23.0` for calculation.

**B. Correlation Comparison (`_compare_correlation`)**
* **What it does:** Measures if the relationships *between* variables were preserved.
* **How it works:** It calculates the Pearson correlation matrix for the numeric columns in the original data and again for the anonymized data. It then reports the **Mean Absolute Difference** between these two matrices.
* **Metric:** A score of `0.0` means the correlations are identical (perfect utility). A score of `0.2` means that, on average, the correlation values (e.g., 0.7 vs 0.5) shifted by 0.2.

**C. Machine Learning (ML) Utility (`_compare_ml_model_performance`)**
* **What it does:** This is the most practical test. It checks if the anonymized data can still be used for a real-world predictive modeling task.
* **How it works:**
    1.  The user defines a simple ML task in the config (e.g., "predict 'Expenditure' using 'Age' and 'Sex'").
    2.  The assessor trains a simple model (like Linear or Logistic Regression) on the **original data** to get a "baseline" score (e.g., 75% accuracy).
    3.  It then trains the *exact same model* on the **anonymized data** to get a new score (e.g., 70% accuracy).
    4.  It reports the final **performance drop** (e.g., a 6.67% drop).

**D. Final Risk Assessment (`_simulate_linkage_attack`)**
* **What it does:** This function re-runs the linkage attack simulation from Module 1, but this time on the **final anonymized data**.
* **Goal:** This is a "pass/fail" check on Module 2. It reports the final `k-anonymity` score and the number of remaining unique records (`k=1`). Ideally, after anonymization, this number should be `0`.

***

## 3. How to Use

The `NSSUtilityAssessor` is called from `main.py` after Module 2 has finished.

**Inputs:**
* `original_df`: The clean, original Pandas DataFrame.
* `anonymized_df`: The anonymized Pandas DataFrame from Module 2.
* `config`: The resolved configuration dictionary.
* `risk_report`: The JSON report from Module 1 (used to get the QI list).

**Output:**
* A comprehensive JSON dictionary (the "Utility Report") containing the results of all four tests.

**Example from `main.py`:**

```Python
# --- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---
logger.info("--- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---")
utility_assessor = NSSUtilityAssessor(resolved_config)
utility_report = utility_assessor.run_utility_analysis(
    analysis_df,  # The original data
    anonymized_df, # The anonymized data
    risk_report['detected_quasi_identifiers'] # QIs from Module 1
)
logger.info(f"Utility Assessment Report: \n{json.dumps(utility_report, indent=2)}")
```


***

### 3. Testing: `test_utility_assessor.py`

You'll need a test file to ensure Module 3 works as expected. Create a `tests` folder inside `module3_utility_assessment` and add the following file. This test file uses the same `conftest.py` fixtures that you already have in the project root (or in `module1_risk_assessment`).

**File: `module3_utility_assessment/tests/test_utility_assessor.py`**

```python
import pytest
import pandas as pd
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor
from conftest import sample_dataframe, test_config # Assumes conftest.py is in root or accessible


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
    anon_df['Age'] = pd.cut(anon_df['Age'], bins=[0, 30, 60], 
                            right=True, labels=['1-30', '31-60'])
                            
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
    assert numeric_age.iloc[0] == 15.5 # Original was 25
    assert numeric_age.iloc[1] == 15.5 # Original was 30
    assert numeric_age.iloc[3] == 45.5 # Original was 45
    
    # Test with a '*' column
    edu_col = anonymized_dataframe['Education_Level']
    numeric_edu = assessor._preprocess_anonymized_column(edu_col)
    assert pd.isna(numeric_edu.iloc[0]) # Original was 'Grad', now '*' -> NaN
    assert pd.isna(numeric_edu.iloc[9]) # Original was 'Grad', now '*' -> NaN


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
    """Tests the final linkage attack on the *anonymized* data."""
    assessor = NSSUtilityAssessor(utility_config)
    # Use the QIs that were *not* generalized for a fair test
    qis = ['Sex', 'District_Code']
    report = assessor._simulate_linkage_attack(anonymized_dataframe, qis)
    
    assert 'final_k_anonymity' in report
    assert 'vulnerable_records_k1' in report
    
    # For (Sex, District_Code), the smallest group is 1
    # (Sex=2, District_Code=101) is unique
    assert report['final_k_anonymity'] == 1
    assert report['vulnerable_records_k1'] > 0
```


***

## 4. Configuration: ingestion_config.yaml

You need to add the new utility_assessment section to your master config file. This section will control what Module 3 measures.

**File:** `module1_risk_assessment/configs/ingestion_config.yaml`

```yaml
# ... (all your existing config for surveys, risk_assessment, etc.)

# ===============================================================
# == MODULE 3 CONFIGURATIONS START HERE ==
# ===============================================================

# Utility Assessment Configuration (For Module 3)
utility_assessment:

  # List of numeric columns to use for statistical and correlation comparisons.
  # These should be columns that are important for analysis.
  # Note: The assessor will auto-convert anonymized ranges (e.g., '20-30')
  # to their midpoints (e.g., 25.0) for these calculations.
  numeric_columns_to_compare:
    - 'Age'
    - 'Household_Size'
    - 'Monthly_Consumer_Expenditure'

  # Configuration for the Machine Learning (ML) utility test.
  # This trains a model on the original and anonymized data to
  # see how much predictive power was lost.
  ml_model:
    # 'regression' (for predicting a number) or 
    # 'classification' (for predicting a category)
    task_type: 'regression'
    
    # The single column you want to predict.
    # Must be a numeric column.
    target_column: 'Monthly_Consumer_Expenditure'
    
    # A list of columns to use as features to predict the target.
    # Can be a mix of numeric and categorical columns.
    feature_columns:
      - 'Age'
      - 'Sex'
      - 'Social_Group'
      - 'Household_Size'
      - 'District_Code'
```


***

## 5. Pipeline Integration: main.py

Finally, you must update your main.py file to import and run this new module.

**File:** `main.py`

```python
"""
Main entry point for the NSS SafeData Pipeline.
This script orchestrates the full pipeline:
1. Data Ingestion (Parsing, Cleaning, Merging)
2. Risk Assessment (Module 1: QI Detection & Risk Scoring)
3. Privacy Enhancement (Module 2: Anonymization)
4. Utility Assessment (Module 3: Utility Measurement)
"""

# ... (all existing imports)

# --- Imports from Module 1 ---
# ... (existing module 1 imports)
from module1_risk_assessment.risk_assessor import NSSRiskAssessor

# --- Imports from Module 2 ---
from module2_privacy_enhancement.privacy_enhancer import NSSPrivacyEnhancer

# --- [NEW] Imports from Module 3 ---
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor

def run_full_pipeline(input_dir: str, output_dir: str, config_path: str, survey_type: str = None):
    """
    Runs the complete data processing, risk assessment, privacy enhancement, and utility pipeline.
    """
    
    # ... (all existing code from STAGE 0 and STAGE 1) ...
    # ... (code to load config, detect survey, parse, clean, merge) ...
    # analysis_df = merger.create_analysis_ready_dataset(merged_df)

    try:
        # --- STAGE 0: CONFIGURATION AND SETUP ---
        # ... (existing code)
        logger.info("Configuration resolved successfully.")

        # --- STAGE 1: DATA INGESTION AND CLEANING ---
        # ... (existing code)
        analysis_df = merger.create_analysis_ready_dataset(merged_df)
        logger.info(f"Data ingestion complete. Analysis dataset shape: {analysis_df.shape}")
        
        # --- STAGE 2: RISK ASSESSMENT (MODULE 1) ---
        logger.info("--- STAGE 2: RISK ASSESSMENT (MODULE 1) ---")
        risk_assessor = NSSRiskAssessor(resolved_config)
        risk_report = risk_assessor.run_risk_analysis(analysis_df)
        logger.info(f"Risk Assessment Report: \n{json.dumps(risk_report, indent=2)}")

        # --- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---
        logger.info("--- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---")
        privacy_enhancer = NSSPrivacyEnhancer(resolved_config, risk_report)
        # Use .copy() to avoid modifying the original dataframe
        anonymized_df = privacy_enhancer.anonymize(analysis_df.copy())
        logger.info("Privacy enhancement complete.")

        # --- [NEW] STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---
        logger.info("--- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---")
        utility_assessor = NSSUtilityAssessor(resolved_config)
        # We need to pass both original and anonymized data, plus the QIs
        utility_report = utility_assessor.run_utility_analysis(
            analysis_df,  # The original, clean data
            anonymized_df, # The newly anonymized data
            risk_report.get('detected_quasi_identifiers', []) # QIs from Module 1
        )
        logger.info(f"Utility Assessment Report: \n{json.dumps(utility_report, indent=2, default=str)}")

        # --- [MODIFIED] STAGE 5: SAVING OUTPUTS ---
        logger.info("--- STAGE 5: SAVING OUTPUTS ---")
        create_output_directory(output_dir)

        # Save the anonymized data
        # ... (existing code to save anonymized_df) ...
        logger.info(f"Anonymized dataset saved to: {output_path}")

        # Save the risk report
        # ... (existing code to save risk_report) ...
        logger.info(f"Risk report saved to: {report_path}")

        # [NEW] Save the utility report
        utility_report_filename = f'utility_report_{survey_type.lower()}.json'
        utility_report_path = os.path.join(output_dir, utility_report_filename)
        with open(utility_report_path, 'w') as f:
            # Use default=str to handle any numpy types (like np.inf)
            json.dump(utility_report, f, indent=4, default=str)
        logger.info(f"Utility report saved to: {utility_report_path}")

        logger.info("==================================================")
        logger.info(" === NSS SAFEDATA PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("==================================================")

    except FileNotFoundError as fnf:
        # ... (existing error handling)
        
# ... (rest of main.py) ...

if __name__ == '__main__':
    main()
```
---