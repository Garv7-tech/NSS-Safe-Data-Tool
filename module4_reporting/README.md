# Module 4: Reporting & Configuration

## 1. Overview

This module is the final stage of the NSS SafeData Pipeline. It serves two critical functions:

1.  **Configuration Management:** It provides a robust validation layer for the entire pipeline's configuration (`ingestion_config.yaml`) using Pydantic models. This ensures all settings are correct before any module runs.
2.  **Report Generation:** It consolidates the outputs from Module 1 (Initial Risk), Module 2 (Anonymization), and Module 3 (Utility Assessment) into comprehensive, user-friendly reports.

---

## 2. Core Components

### `config_models.py`
This file contains a set of nested **Pydantic models** (e.g., `MasterConfig`, `SurveyConfig`, `PrivacyEnhancementConfig`).

* **Purpose:** To load, parse, and validate the *entire* `ingestion_config.yaml` file.
* **Function:** If the config file is missing a required field (like `target_k`) or has an invalid value (like `max_info_loss = 2.0`), Pydantic will raise a clear validation error. This makes the entire pipeline robust and prevents errors deep in the workflow. It is the "single source of truth" for all settings.

### `report_generator.py`
This file contains the main `NSSReportGenerator` class, which does the reporting.

#### Key Responsibilities:

1.  **`generate_json_report()`**
    * **What it does:** Takes the Python dictionaries from Modules 1, 2, and 3 as input.
    * **Output:** Produces a single, structured **JSON report**. This JSON is designed to be directly consumed by a web frontend to build a results dashboard. It includes a top-level `summary` for key metrics, followed by the detailed reports for each module.

2.  **`save_pdf_report()`**
    * **What it does:** Takes the JSON report data and generates a human-readable **PDF report**.
    * **Library:** Uses `ReportLab` (a standard Python library for PDF creation).
    * **Output:** A downloadable `.pdf` file that presents the summary, risk scores, and utility metrics in a formatted way.

---

## 3. Example JSON Report Structure

The `generate_json_report()` method produces a structure similar to this, which is ideal for a frontend UI:

```json
{
  "summary": {
    "pipeline_status": "Success",
    "survey_type": "PLFS",
    "survey_name": "Periodic Labour Force Survey",
    "original_records": 11,
    "anonymization_strategy": "k_anonymity",
    "final_k_anonymity": 3,
    "vulnerable_records_remaining": 0,
    "final_re_id_risk_percent": 0.0,
    "ml_utility_score": 0.85,
    "ml_performance_drop_percent": 10.2
  },
  "initial_risk_assessment": {
    "detected_quasi_identifiers": ["Age", "Sex", "District_Code", "Education_Level"],
    "risk_metrics": {
      "k_anonymity": 1,
      "total_records": 11
    },
    "linkage_attack_simulation": {
      "vulnerable_records_k1": 5,
      "reidentification_rate_percent": 45.45
    }
  },
  "privacy_enhancement_details": {
    "anonymization_strategy": "k_anonymity",
    "goals_met": true,
    "final_k": 3,
    "final_info_loss_ncp": 0.22
  },
  "utility_assessment": {
    "statistical_comparison": {
      "Age": {
        "original_mean": 38.8,
        "anonymized_mean": 37.5,
        "mean_diff_percent": -3.3
      }
    },
    "correlation_comparison": {
      "mean_absolute_correlation_difference": 0.08
    },
    "ml_model_comparison": {
      "task_type": "regression",
      "original_score (r2_score)": 0.95,
      "anonymized_score (r2_score)": 0.85,
      "performance_drop_percent": 10.2
    },
    "anonymized_data_risk": {
      "final_k_anonymity": 3,
      "vulnerable_records_k1": 0,
      "reidentification_rate_percent": 0.0
    }
  }
}