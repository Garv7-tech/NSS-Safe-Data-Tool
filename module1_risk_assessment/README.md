# Module 1: Risk Assessment

## 1. Overview

This module serves as the diagnostic core of the NSS SafeData Pipeline. Its primary responsibility is to analyze the clean, merged dataset *before* any anonymization is applied. It quantifies the inherent privacy risks by automatically identifying potential Quasi-Identifiers (QIs) and calculating standard privacy metrics.

The entire module is designed to be **survey-agnostic**, meaning all logic is driven by the central `ingestion_config.yaml` file, not hardcoded rules.

---

## 2. Core Components

The logic for this module is encapsulated within a single primary class located in `risk_assessor.py`.

### `NSSRiskAssessor`

This class is the main engine for all risk assessment tasks.

#### Key Responsibilities:

**A. Robust Quasi-Identifier (QI) Detection**

The most critical feature is its intelligent, **3-Layer Hybrid Approach** to detecting QIs, ensuring a comprehensive and accurate identification of risky columns.

* **Layer 1: Config-Based Identification:**
    * It first reads a baseline list of potential QIs from the `risk_assessment.quasi_identifier_candidates` section in `ingestion_config.yaml`. This allows domain-specific knowledge to be easily injected.

* **Layer 2: Semantic Analysis:**
    * The method `_is_semantic_qi_candidate` analyzes column names for common patterns (e.g., 'age', 'district', 'code'). This helps identify potential QIs even if they are not explicitly listed in the config.

* **Layer 3: Statistical Analysis:**
    * The method `_get_statistical_qi_score` scores each column based on its data distribution. It identifies columns that are strong QI candidates by checking their cardinality (number of unique values). Columns that are too unique (like a primary key) or not unique at all (only one value) are scored low.

**B. Risk Metrics Calculation**

* **k-Anonymity (`calculate_k_anonymity`):** This function calculates the k-anonymity of the dataset for a given set of QIs. It works by grouping the data by the QIs and finding the size of the smallest group. A `k` value of 1 means at least one individual is unique and highly re-identifiable.

**C. Linkage Attack Simulation**

* **`simulate_linkage_attack`:** To provide a practical measure of risk, this function simulates a real-world privacy attack. It creates a mock "attacker's dataset" from a sample of the original data and attempts to link it back to the unique and vulnerable records (where k=1). The output is a **Re-identification Rate**, which is a powerful metric to demonstrate the need for anonymization.

---

## 3. Configuration

All operations of this module are controlled via the `ingestion_config.yaml` file.

**File:** `configs/ingestion_config.yaml`

```yaml
# Risk Assessment Configuration (For Module 1)
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
quasi_identifier_candidates: This section provides the baseline list for the QI detection algorithm. common applies to all surveys, while survey_specific allows for survey-dependent QIs.

4. How to Use
The NSSRiskAssessor is designed to be called from the main pipeline (main.py) after the data has been ingested and cleaned.

Inputs:

df: The clean Pandas DataFrame.

config: The resolved configuration dictionary for the specific survey.

Output:

A JSON dictionary (the "Risk Report") containing the list of detected QIs and calculated risk scores.

Example from main.py:

```Python

# --- STAGE 2: RISK ASSESSMENT (MODULE 1) ---
logger.info("--- STAGE 2: RISK ASSESSMENT (MODULE 1) ---")
risk_assessor = NSSRiskAssessor(resolved_config)
risk_report = risk_assessor.run_risk_analysis(analysis_df)
logger.info(f"Risk Assessment Report: \n{json.dumps(risk_report, indent=2)}")
```

### 5. Testing
Unit tests for this module are located in module1_risk_assessment/tests/.

conftest.py: Contains shared fixtures, such as a sample DataFrame and configuration, used across all tests.

test_risk_assessor.py: Contains specific unit tests for the NSSRiskAssessor class. It validates the QI detection logic, k-anonymity calculation, and edge cases like empty dataframes.

To run the tests, navigate to the project's root directory and execute:

```Bash
pytest -v
```

