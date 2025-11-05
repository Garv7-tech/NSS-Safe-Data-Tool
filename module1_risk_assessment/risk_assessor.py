import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskAssessor:
    """
    Assesses the re-identification risk of a given dataset.
    
    New in this version:
    - Includes `simulate_linkage_attack` to fulfill PS-1.pdf requirement.
    """
    def __init__(self, analysis_df: pd.DataFrame, survey_config: Dict[str, Any]):
        self.df = analysis_df
        self.config = survey_config
        # Get QIs from config, default to empty list if not specified
        self.quasi_identifiers: List[str] = self.config.get('quasi_identifiers', [])
        self.report: Dict[str, Any] = {
            'risk_metrics': {},
            'attack_simulation': {}
        }

    def _calculate_k_anonymity(self) -> Dict[str, Any]:
        """Calculates k-anonymity based on the configured quasi-identifiers."""
        if not self.quasi_identifiers:
            logger.warning("No quasi-identifiers specified in config. Skipping k-anonymity calculation.")
            return {
                'status': 'skipped', 
                'reason': 'No quasi-identifiers specified'
            }
            
        logger.info(f"Calculating k-anonymity using QIs: {self.quasi_identifiers}")
        
        # Check if all QIs are in the dataframe
        missing_qis = [qi for qi in self.quasi_identifiers if qi not in self.df.columns]
        if missing_qis:
            logger.error(f"Missing QIs in DataFrame, cannot assess risk: {missing_qis}")
            return {'status': 'failed', 'error': f'Missing QIs: {missing_qis}'}

        # Calculate equivalence class sizes
        try:
            equiv_class_sizes = self.df.groupby(self.quasi_identifiers).size()
            min_k = equiv_class_sizes.min()
            
            # Count records at risk (k=1)
            vulnerable_records_count = int((equiv_class_sizes == 1).sum())
            total_records = len(self.df)
            vulnerable_percentage = (vulnerable_records_count / total_records) * 100 if total_records > 0 else 0

            logger.info(f"Minimum k-anonymity: {min_k}")
            logger.info(f"Vulnerable records (k=1): {vulnerable_records_count} ({vulnerable_percentage:.2f}%)")

            return {
                'status': 'success',
                'min_k': int(min_k),
                'vulnerable_records_count': vulnerable_records_count,
                'vulnerable_records_percentage': round(vulnerable_percentage, 2),
                'total_records': total_records,
                'quasi_identifiers_used': self.quasi_identifiers
            }
        except Exception as e:
            logger.error(f"Failed to calculate k-anonymity: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def simulate_linkage_attack(self, ground_truth_file: Path) -> Dict[str, Any]:
        """
        Simulates a linkage attack using an external "ground truth" file.
        This file is assumed to have QIs and one or more true identifiers
        (e.g., Name, PII).
        """
        logger.info(f"--- Starting Linkage Attack Simulation ---")
        logger.info(f"Loading ground truth file from: {ground_truth_file}")
        
        try:
            # We assume the ground truth file is a CSV
            ground_truth_df = pd.read_csv(ground_truth_file)
        except Exception as e:
            logger.error(f"Failed to load ground truth file: {e}")
            return {'status': 'failed', 'error': str(e)}

        # Identify QIs present in *both* datasets
        common_qis = [qi for qi in self.quasi_identifiers if qi in ground_truth_df.columns]
        if not common_qis:
            logger.warning("No common QIs found between survey data and ground truth file. Skipping attack.")
            return {'status': 'skipped', 'reason': 'No common QIs'}

        logger.info(f"Attacking using common QIs: {common_qis}")

        # Perform the linkage attack (a simple inner merge on the QIs)
        try:
            # Drop duplicates from ground truth to avoid ambiguity
            attacker_df = ground_truth_df[common_qis].drop_duplicates()
            
            # Count how many unique groups in our data match the attacker's data
            merged = self.df[common_qis].merge(attacker_df, on=common_qis, how='inner')
            
            # Count records successfully re-identified
            # This counts unique *individuals* in the original data that were linked
            linked_record_count = merged.drop_duplicates().shape[0]
            total_records = len(self.df)
            linked_percentage = (linked_record_count / total_records) * 100 if total_records > 0 else 0

            logger.info(f"Attack successful: {linked_record_count} records re-identified.")

            return {
                'status': 'success',
                'common_qis_used': common_qis,
                'linked_record_count': int(linked_record_count),
                'total_records': total_records,
                're_identification_percentage': round(linked_percentage, 2)
            }
        except Exception as e:
            logger.error(f"Linkage attack simulation failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def assess_risk(self, ground_truth_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Runs all risk assessment tasks.
        Now includes the optional linkage attack.
        """
        logger.info("--- STAGE 2: RISK ASSESSMENT (MODULE 1) ---")
        
        # 1. Calculate k-anonymity (original functionality)
        self.report['risk_metrics'] = self._calculate_k_anonymity()
        
        # 2. Simulate linkage attack (new functionality)
        if ground_truth_file:
            if ground_truth_file.exists():
                self.report['attack_simulation'] = self.simulate_linkage_attack(ground_truth_file)
            else:
                logger.warning(f"Ground truth file specified but not found: {ground_truth_file}")
                self.report['attack_simulation'] = {'status': 'skipped', 'reason': 'File not found'}
        else:
            logger.info("No ground truth file provided, skipping linkage attack simulation.")
            self.report['attack_simulation'] = {'status': 'skipped', 'reason': 'Not provided'}

        logger.info("Risk assessment complete.")
        return self.report