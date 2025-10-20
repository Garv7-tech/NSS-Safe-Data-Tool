# module1_risk_assessment/risk_assessor.py

import pandas as pd
from typing import Dict, List, Any, Set
import logging
from .utils import setup_logging
import re # Regular expressions for semantic analysis

class NSSRiskAssessor:
    """
    Performs a comprehensive, multi-layered risk analysis on NSS data.
    - Robust QI Detection using a hybrid approach.
    - Calculation of various risk metrics.
    - Simulation of privacy attacks.
    """

    def __init__(self, config: Dict):
        """
        Initialize the assessor with the resolved survey-specific configuration.
        """
        self.config = config
        self.logger = setup_logging(__name__)
        # Direct Identifiers ko hamesha ignore karna hai
        self.DIRECT_IDENTIFIERS = {'Person_Serial_No', 'Sample_Household_Number', 'HHID'}

    # LAYER 2: Semantic Analysis
    def _is_semantic_qi_candidate(self, column_name: str) -> bool:
        """
        Analyzes the column name to guess if it's a potential QI.
        यह Column के नाम से अंदाज़ा लगाता है कि क्या वह QI हो सकता है।
        """
        # Common QI patterns in names (case-insensitive)
        qi_name_patterns = [
            'age', 'sex', 'gender', 'district', 'state', 'region', 'code',
            'status', 'level', 'sector', 'type', 'group', 'religion'
        ]
        # Check if any pattern exists in the column name
        for pattern in qi_name_patterns:
            if re.search(pattern, column_name, re.IGNORECASE):
                return True
        return False

    # LAYER 3: Statistical Analysis
    def _get_statistical_qi_score(self, column: pd.Series) -> float:
        """
        Calculates a "QI-ness" score for a column based on its data.
        यह Column के डेटा के आधार पर 0 से 1 के बीच एक "QI स्कोर" देता है।
        """
        total_rows = len(column)
        unique_values = column.nunique()

        # Rule 1: Ignore columns that are unique keys or have only one value.
        if unique_values <= 1 or unique_values >= total_rows * 0.95:
            return 0.0

        # Rule 2: High score for categorical data with low to medium cardinality.
        # (e.g., State, District, Social_Group)
        if pd.api.types.is_categorical_dtype(column.dtype) or pd.api.types.is_object_dtype(column.dtype):
            # 2 to 100 categories is a strong indicator of a QI
            if 2 < unique_values < 100:
                return 0.9
            else:
                return 0.5 # Still a candidate, but weaker

        # Rule 3: Score for numerical data based on uniqueness ratio.
        # (e.g., Age, Household_Size)
        if pd.api.types.is_numeric_dtype(column.dtype):
            uniqueness_ratio = unique_values / total_rows
            # Columns with 1% to 30% unique values are strong candidates.
            if 0.01 < uniqueness_ratio < 0.30:
                return 0.8
            # Weaker candidates
            elif 0.30 <= uniqueness_ratio < 0.60:
                return 0.4

        return 0.0 # Default: Not a QI

    # Main Detection Method
    def detect_quasi_identifiers(self, df: pd.DataFrame, score_threshold: float = 0.35) -> List[str]:
        """
        Detects QIs using the 3-layer hybrid approach.
        """
        self.logger.info("--- Starting Robust QI Detection (3-Layer Hybrid Approach) ---")
        final_qi_set: Set[str] = set()

        # LAYER 1: Get candidates from ingestion_config.yaml
        risk_config = self.config.get('risk_assessment', {})
        qi_candidates_config = risk_config.get('quasi_identifier_candidates', {})
        common_qis = qi_candidates_config.get('common', [])
        survey_type = self.config.get('survey_type_detected')
        survey_specific_qis = qi_candidates_config.get('survey_specific', {}).get(survey_type, [])
        config_based_qis = {col for col in (common_qis + survey_specific_qis) if col in df.columns}
        final_qi_set.update(config_based_qis)
        self.logger.info(f"[Layer 1] Found {len(config_based_qis)} QIs from config: {list(config_based_qis)}")

        # Layers 2 & 3: Semantic and Statistical Analysis
        self.logger.info(f"[Layers 2 & 3] Analyzing all columns with a score threshold of {score_threshold}...")
        for col_name in df.columns:
            if col_name in final_qi_set or col_name in self.DIRECT_IDENTIFIERS:
                continue # Skip if already found or it's a direct ID

            # Layer 2 Check
            is_semantic_candidate = self._is_semantic_qi_candidate(col_name)

            # Layer 3 Check
            statistical_score = self._get_statistical_qi_score(df[col_name])

            # Combine scores - Semantic match gives a boost
            final_score = statistical_score
            if is_semantic_candidate:
                final_score += 0.2 # Bonus points for a good name

            if final_score > score_threshold:
                final_qi_set.add(col_name)
                self.logger.info(f"  -> Detected '{col_name}' | Semantic Match: {is_semantic_candidate} | Final Score: {final_score:.2f}")

        detected_qis = sorted(list(final_qi_set))
        self.logger.info(f"--- Total QIs Detected: {len(detected_qis)} -> {detected_qis} ---")
        return detected_qis

    def calculate_k_anonymity(self, df: pd.DataFrame, qi_columns: List[str]) -> int:
        # (यह function पहले जैसा ही रहेगा)
        if df.empty or not qi_columns: return 0
        return int(df.groupby(qi_columns).size().min())

    def run_risk_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the full risk analysis pipeline.
        """
        self.logger.info("--- Starting Initial Risk Analysis ---")
        detected_qis = self.detect_quasi_identifiers(df)
        k_anonymity_score = self.calculate_k_anonymity(df, detected_qis)

        risk_report = {
            'detected_quasi_identifiers': detected_qis,
            'risk_metrics': {
                'k_anonymity': k_anonymity_score
            }
        }
        self.logger.info(f"Risk analysis report: {risk_report}")
        return risk_report
    
    def simulate_linkage_attack(self, df: pd.DataFrame, qi_columns: List[str]) -> Dict[str, Any]:
        """
        Simulates a linkage attack to calculate the re-identification risk.
        """
        self.logger.info(f"--- Starting Linkage Attack Simulation on QIs: {qi_columns} ---")

        if not qi_columns:
            self.logger.warning("No QIs provided for attack simulation.")
            return {}

        # Step 1: Create a fake "Attacker's Dataset".
        # We take a random 20% sample from the original data to act as the public dataset.
        attacker_df = df[qi_columns].sample(frac=0.2, random_state=42).drop_duplicates()
        attacker_df['attacker_identifier'] = [f'Person_{i}' for i in range(len(attacker_df))]
        self.logger.info(f"Created a fake Attacker's Dataset with {len(attacker_df)} unique records.")

        # Step 2: Perform the linkage (merge).
        # We count how many groups in the original data are unique (k=1).
        # These are the most vulnerable individuals.
        equivalence_classes = df.groupby(qi_columns).size().reset_index(name='k_value')
        vulnerable_groups = equivalence_classes[equivalence_classes['k_value'] == 1]

        # The attack links the attacker's data with these vulnerable groups.
        successful_links = pd.merge(vulnerable_groups, attacker_df, on=qi_columns)
        
        # Step 3: Measure the success.
        total_records = len(df)
        vulnerable_records = len(vulnerable_groups)
        reidentified_records = len(successful_links)

        # Calculate re-identification rate (from the attacker's perspective)
        if not attacker_df.empty:
            reidentification_rate = (reidentified_records / len(attacker_df)) * 100
        else:
            reidentification_rate = 0

        self.logger.info(f"Total Records in original data: {total_records}")
        self.logger.info(f"Records that are unique (k=1) and vulnerable: {vulnerable_records}")
        self.logger.info(f"Records successfully re-identified in the attack: {reidentified_records}")
        self.logger.info(f"Re-identification Rate: {reidentification_rate:.2f}%")

        simulation_report = {
            'attack_qis': qi_columns,
            'total_records': total_records,
            'vulnerable_records_k1': vulnerable_records,
            'reidentified_records': reidentified_records,
            'reidentification_rate_percent': round(reidentification_rate, 2)
        }

        return simulation_report