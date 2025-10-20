# Module 2: Privacy Enhancement

## 1. Overview

This module is the heart of the NSS SafeData Pipeline, responsible for performing the actual anonymization of the data. Its core design philosophy is to be **goal-seeking, not rule-based**. Instead of following a fixed set of instructions, this module takes a set of privacy goals (e.g., "achieve k=10 with less than 20% information loss") and intelligently applies anonymization techniques to meet them.

It features a hybrid strategy system, allowing it to switch between different anonymization techniques based on the user's requirements.

---

## 2. Core Components

The logic for this module is encapsulated within the `NSSPrivacyEnhancer` class, located in `privacy_enhancer.py`.

### `NSSPrivacyEnhancer`

This class acts as an intelligent anonymization engine. It takes the clean data and the risk report from Module 1 and produces an anonymized dataset.

#### Key Responsibilities:

**A. Hybrid Anonymization Strategy**

The engine can employ different strategies based on the `privacy_strategy` setting in the configuration.

1.  **k-Anonymity (Default Strategy):**
    * **Goal:** To ensure that any individual in the dataset cannot be distinguished from at least `k-1` other individuals based on their Quasi-Identifiers (QIs).
    * **Algorithm (`_generalize_k_anonymity`):** It uses a **greedy partitioning algorithm**. The data is iteratively grouped into small clusters, each containing at least `k` records. The values within each cluster are then **generalized**.
        * **Numeric Generalization:** `Age` values `21, 23, 25` might become the range `'21-25'`.
        * **Categorical Generalization:** `District_Code` values `101, 102` might become `'101|102'`.

2.  **Synthetic Data Generation (SDG):**
    * **Goal:** To be used when generalization causes too much information loss, or when a higher degree of privacy is needed.
    * **Algorithm (`_run_synthetic_data_generation`):** This strategy uses the **`sdv` (Synthetic Data Vault)** library. It learns the statistical properties and correlations from the original data and then generates a brand new, artificial dataset that has the same statistical characteristics but contains no real records.

**B. Goal-Seeking Optimization Loop**

For the `k_anonymity` strategy, the engine operates in a feedback loop (`_run_k_anonymity_optimization`):
1.  **Apply:** It applies the generalization algorithm to the data.
2.  **Measure:** It calculates the resulting `k-anonymity` and `Information Loss` of the anonymized data.
3.  **Compare:** It checks if the measured values meet the `target_k` and `max_info_loss` goals defined in the config.
4.  **Result:** If the goals are met, it returns the anonymized data. If not, it logs an error, indicating that the goals might be too strict for the given data.

**C. Information Loss Metric**

* **Normalized Certainty Penalty (NCP) (`_calculate_normalized_certainty_penalty`):** To measure data utility, the engine uses NCP, a standard academic metric. It quantifies how much information was lost during generalization. A score of 0 means no loss, while a score of 1 means total loss of data.

---

## 3. Configuration

The behavior of this module is controlled via the `privacy_enhancement` section in `ingestion_config.yaml`.

**File:** `configs/ingestion_config.yaml`

```yaml
# Privacy Enhancement Configuration (For Module 2)
privacy_enhancement:
  # Choose the strategy: 'k_anonymity', 'synthetic_data'
  privacy_strategy: 'k_anonymity'

  # Goals for the anonymization engine
  goals:
    k_anonymity:
      target_k: 10
      max_info_loss: 0.25 # Max 25% info loss allowed (using NCP metric)
privacy_strategy: Determines which anonymization method to use.

target_k: The minimum k value the engine must achieve.

max_info_loss: The maximum allowed NCP score. The anonymization will be considered a failure if the information loss exceeds this threshold.

4. How to Use
The NSSPrivacyEnhancer is designed to be called directly after Module 1 completes.

Inputs:

df: The clean Pandas DataFrame (from the data ingestion stage).

config: The resolved configuration dictionary.

risk_report: The JSON output from NSSRiskAssessor (from Module 1), which contains the list of detected QIs.

Output:

A new, anonymized Pandas DataFrame.

Example from main.py:

Python

# --- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---
logger.info("--- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---")
privacy_enhancer = NSSPrivacyEnhancer(resolved_config, risk_report)
anonymized_df = privacy_enhancer.anonymize(analysis_df.copy())
5. Testing
Unit tests for this module are located in module2_privacy_enhancement/tests/.

test_privacy_enhancer.py: Contains unit tests for the NSSPrivacyEnhancer class. Key tests include:

Verifying that the target k is achieved.

Ensuring the system correctly identifies when goals cannot be met (e.g., information loss is too high).

Validating the NCP calculation.

Testing the synthetic data generation strategy.

To run the tests, navigate to the project's root directory and execute:

```Bash
pytest -v
```