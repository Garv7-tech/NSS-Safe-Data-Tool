# Module 1: Risk Assessment

## 1. Overview

This module serves as the diagnostic core of the NSS SafeData Pipeline. Its primary responsibility is to analyze the clean, merged dataset *before* any anonymization is applied. It quantifies the inherent privacy risks by automatically identifying potential Quasi-Identifiers (QIs) and calculating standard privacy metrics.

This module is designed to be **config-driven**. All its logic (like *which* columns to check) is controlled by the `ingestion_config.yaml` file, not hardcoded.

**New functionality** based on the `PS-1.pdf` problem statement has been included to simulate a direct **linkage attack** using an auxiliary "ground truth" file, providing a true measure of re-identification risk.

---

## 2. Core Components

The logic for this module is encapsulated within the `NSSRiskAssessor` class.

### `NSSRiskAssessor`

This class is the main engine for all risk assessment tasks.

#### Key Responsibilities:

**A. Robust Quasi-Identifier (QI) Detection**
* **`detect_quasi_identifiers`:** This method is the core of the risk detection. It uses a **3-Layer Hybrid Approach** to find risky columns:
    * **Layer 1: Config-Based:** It reads a baseline list of potential QIs from the `risk_assessment.quasi_identifier_candidates` section in `ingestion_config.yaml`. This allows domain-specific knowledge to be injected.
    * **Layer 2: Semantic Analysis (`_is_semantic_qi_candidate`):** Analyzes column names for common patterns (e.g., 'age', 'district', 'code', 'level') to find QIs not listed in the config.
    * **Layer 3: Statistical Analysis (`_get_statistical_qi_score`):** Scores each column based on its data distribution (cardinality). It gives a high score to columns that are neither too unique (like an ID) nor not unique at all (like a constant value).

**B. Risk Metrics Calculation**
* **`calculate_k_anonymity`:** This function calculates the k-anonymity of the dataset for a given set of QIs. It works by grouping the data by the QIs and finding the size of the smallest group.
* **`k=1` (High Risk):** A `k` value of 1 means at least one individual (or group) is unique based on these QIs and is highly re-identifiable. The report also counts *how many* such records exist.

**C. Linkage Attack Simulation (Fulfills PS-1.pdf Requirement)**
* **`simulate_linkage_attack`:** This method simulates a realistic privacy attack as requested in the problem statement.
* **How it works:** It takes an optional "Auxiliary 'Ground Truth' File" (e.g., a public voter list) provided via the `--ground-truth-file` argument in `main.py`. It then attempts to **merge (link)** this public file with the survey data using the detected QIs.
* **Output:** It produces a concrete **Re-identification Rate**, (e.g., "15.2% of unique groups were successfully linked"), quantifying the *actual* re-identification risk.

---

## 3. Configuration

All operations of this module are controlled via the `ingestion_config.yaml` file.

**File: `configs/ingestion_config.yaml`**
```yaml
risk_assessment:
  quasi_identifier_candidates:
    common:
      - 'Age'
      - 'Sex'
      - 'Marital_Status'
      - 'State_Ut_Code'
      - 'District_Code'
    survey_specific:
      PLFS:
        - 'General_Education_Level'
      HCES:
        - 'Religion'
```

## 4. How to Use

The `NSSRiskAssessor` is called from `main.py` after the data has been ingested.

### Inputs:
* `df`: The clean Pandas DataFrame.
* `config`: The resolved configuration dictionary for the specific survey.
* `ground_truth_file` (Optional): The path to the auxiliary file for the linkage attack.

### Output:
A JSON dictionary (the "Risk Report") containing:
* `detected_quasi_identifiers`: The final list of QIs found.
* `risk_metrics`: k-anonymity score, count of vulnerable (k=1) records, etc.
* `linkage_attack_simulation`: The results of the linkage attack (e.g., `reidentification_rate_percent`).

### Example from `main.py`:
```python
# --- STAGE 2: RISK ASSESSMENT (MODULE 1) ---
risk_assessor = NSSRiskAssessor(resolved_config)
risk_report = risk_assessor.run_risk_analysis(
    analysis_df,
    ground_truth_file=args.ground_truth_file # New argument
)
```

## 5. Testing

Unit tests are in `module1_risk_assessment/tests/test_risk_assessor.py`. They validate:
* The 3-layer QI detection logic.
* The k-anonymity calculation.
* The new `simulate_linkage_attack` function using a mocked "attacker" file.

The main run_risk_analysis function.


---

### 3. Module 1 Test: `module1_risk_assessment/tests/test_risk_assessor.py` (Updated)

This is the **complete test file** that replaces your old one. It is updated to test the new linkage attack feature and use the new `conftest.py` fixtures.

**File: `module1_risk_assessment/tests/test_risk_assessor.py` (Updated & Complete)**
```python
import pytest
import pandas as pd
from module1_risk_assessment.risk_assessor import NSSRiskAssessor
from conftest import sample_dataframe, full_master_config_dict # Import fixtures
from unittest.mock import patch
from pathlib import Path

@pytest.fixture
def resolved_plfs_config(full_master_config_dict):
    """
    A fixture to simulate the 'resolved_config' for PLFS
    that main.py would normally create.
    """
    # This simulates the output of NSSConfigResolver
    config = {
        'survey_type_detected': 'PLFS',
        'risk_assessment': full_master_config_dict['risk_assessment'],
        'analysis_columns': full_master_config_dict['analysis_columns']
    }
    return config

def test_detect_quasi_identifiers(sample_dataframe, resolved_plfs_config):
    """Tests the 3-layer hybrid QI detection."""
    assessor = NSSRiskAssessor(resolved_plfs_config)
    detected_qis = assessor.detect_quasi_identifiers(sample_dataframe)
    
    # QIs from config: 'Age', 'Sex', 'District_Code', 'Education_Level', 'Social_Group'
    # 'Person_Serial_No' should be in 'DIRECT_IDENTIFIERS' and ignored.
    # 'Monthly_Consumer_Expenditure' and 'Household_Size' are numeric and
    # will be scored by statistical analysis.
    
    expected_qis = [
        'Age', 
        'Sex', 
        'District_Code', 
        'Education_Level', 
        'Social_Group',
        'Household_Size' # This should be picked up by statistical analysis
    ]
    
    assert sorted(detected_qis) == sorted(expected_qis)
    assert 'Person_Serial_No' not in detected_qis
    assert 'Monthly_Consumer_Expenditure' not in detected_qis # Too many unique values

def test_calculate_k_anonymity(sample_dataframe, resolved_plfs_config):
    """Tests the k-anonymity calculation."""
    assessor = NSSRiskAssessor(resolved_plfs_config)
    
    # Test with QIs that have a k=1 record
    # (Age=28, Sex=1, District_Code=101) is unique
    qi_columns_k1 = ['Age', 'Sex', 'District_Code']
    assert assessor.calculate_k_anonymity(sample_dataframe, qi_columns_k1) == 1

    # Test with QIs that have a k > 1
    # Smallest group for 'Sex' is 5 (Sex=2)
    qi_columns_k5 = ['Sex']
    assert assessor.calculate_k_anonymity(sample_dataframe, qi_columns_k5) == 5

def test_k_anonymity_on_empty_dataframe(resolved_plfs_config):
    """Tests that k-anonymity on an empty dataframe is 0."""
    assessor = NSSRiskAssessor(resolved_plfs_config)
    empty_df = pd.DataFrame({'Age': [], 'Sex': []})
    assert assessor.calculate_k_anonymity(empty_df, ['Age', 'Sex']) == 0

@patch('pandas.read_csv')
def test_simulate_linkage_attack(mock_read_csv, sample_dataframe, resolved_plfs_config):
    """Tests the linkage attack simulation using a mock 'ground truth' file."""
    
    # 1. Create the mock attacker's data (the "ground truth" file)
    attacker_data = {
        'Age': [28, 60, 45, 99], # Person 1, 2, 3 (match) and 4 (no match)
        'Sex': [1, 1, 2, 1],
        'District_Code': [101, 104, 103, 999],
        'Name': ['Alice', 'Bob', 'Carol', 'David'] # The PII
    }
    mock_attacker_df = pd.DataFrame(attacker_data)
    mock_read_csv.return_value = mock_attacker_df
    
    assessor = NSSRiskAssessor(resolved_plfs_config)
    qis = ['Age', 'Sex', 'District_Code'] # QIs for the attack
    
    # 3. Run simulation
    report = assessor.simulate_linkage_attack(sample_dataframe, qis, Path('fake/path.csv'))

    # 4. Check results
    assert report['status'] == 'success'
    assert report['common_qis_used'] == ['Age', 'Sex', 'District_Code']
    
    # Total unique (Age, Sex, District_Code) groups in sample_dataframe is 7
    assert report['total_unique_groups_in_survey'] == 7
    
    # Attacker's unique groups: (28, 1, 101), (60, 1, 104), (45, 2, 103), (99, 1, 999)
    # The first 3 of these match unique groups in the survey data.
    assert report['reidentified_unique_groups'] == 3
    
    # 3 / 7 = 42.857...
    assert report['reidentification_rate_percent'] == pytest.approx(42.86)

@patch('pandas.read_csv')
def test_run_risk_analysis_with_attack(mock_read_csv, sample_dataframe, resolved_plfs_config):
    """Tests the main run_risk_analysis function call *with* the ground truth file."""
    
    # Mock attacker data
    attacker_data = {'Age': [28], 'Sex': [1], 'District_Code': [101], 'Name': ['Test']}
    mock_attacker_df = pd.DataFrame(attacker_data)
    mock_read_csv.return_value = mock_attacker_df
    
    # Mock Path.exists() to return True
    with patch.object(Path, 'exists', return_value=True):
        assessor = NSSRiskAssessor(resolved_plfs_config)
        report = assessor.run_risk_analysis(sample_dataframe, ground_truth_file=Path('fake/path.csv'))

    # Check QI detection
    assert 'Age' in report['detected_quasi_identifiers']
    assert 'Sex' in report['detected_quasi_identifiers']
    
    # Check risk metrics (k=1, 5 vulnerable records)
    assert report['risk_metrics']['k_anonymity'] == 1
    assert report['risk_metrics']['vulnerable_records_k1'] == 5

    # Check that attack simulation *ran* and found a match
    assert report['linkage_attack_simulation']['status'] == 'success'
    assert report['linkage_attack_simulation']['reidentified_unique_groups'] == 1

def test_run_risk_analysis_no_attack(sample_dataframe, resolved_plfs_config):
    """Tests the main run_risk_analysis function call *without* the ground truth file."""
    
    assessor = NSSRiskAssessor(resolved_plfs_config)
    
    # Run *without* the ground truth file
    report = assessor.run_risk_analysis(sample_dataframe, ground_truth_file=None)
    
    # Check risk metrics
    assert report['risk_metrics']['k_anonymity'] == 1
    assert report['risk_metrics']['vulnerable_records_k1'] == 5
    
    # Check that attack simulation was *skipped*
    assert report['linkage_attack_simulation']['status'] == 'skipped'
    assert report['linkage_attack_simulation']['reason'] == 'Not provided'