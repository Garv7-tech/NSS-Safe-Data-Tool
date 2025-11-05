"""
Main entry point for the NSS SafeData Pipeline.
This script orchestrates the full pipeline:
1. Data Ingestion (Parsing, Cleaning, Merging)
2. Risk Assessment (Module 1: QI Detection & Risk Scoring)
3. Privacy Enhancement (Module 2: Anonymization)
4. Utility Assessment (Module 3: Utility Measurement)
"""

import argparse
import os
import json
import yaml
from pathlib import Path
import pandas as pd
import logging

# --- Imports from Module 1 ---
from module1_risk_assessment.data_ingestion.file_parser import NSSCSVParser
from module1_risk_assessment.data_ingestion.data_cleaner import NSSDataCleaner
from module1_risk_assessment.data_ingestion.data_merger import NSSDataMerger
from module1_risk_assessment.data_ingestion.survey_detector import NSSConfigResolver, NSSSurveyDetector
from module1_risk_assessment.data_ingestion.utils import setup_logging, get_memory_usage, create_output_directory
from module1_risk_assessment.risk_assessor import NSSRiskAssessor

# --- Imports from Module 2 ---
from module2_privacy_enhancement.privacy_enhancer import NSSPrivacyEnhancer

# --- Imports from Module 3 ---
from module3_utility_assessment.utility_assessor import NSSUtilityAssessor


def run_full_pipeline(input_dir: str, output_dir: str, config_path: str, survey_type: str = None):
    """
    Runs the complete data processing, risk assessment, privacy enhancement, and utility pipeline.
    """

    # Use the logging setup from the utils
    logger = setup_logging(__name__)
    logger.info("==================================================")
    logger.info("      === STARTING NSS SAFEDATA PIPELINE ===      ")
    logger.info("==================================================")

    try:
        # --- STAGE 0: CONFIGURATION AND SETUP ---
        logger.info("--- STAGE 0: CONFIGURATION AND SETUP ---")
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration file from {config_path}: {e}")
            return

        # Auto-detect survey type if not provided
        if not survey_type:
            logger.info("No survey type provided, attempting auto-detection...")
            detector = NSSSurveyDetector(full_config)
            survey_type = detector.detect_survey_type(input_dir)
        logger.info(f"Using Survey Type: {survey_type}")

        # Resolve the configuration for the detected survey
        resolver = NSSConfigResolver(full_config)
        resolved_config = resolver.resolve_config_for_survey(survey_type)
        resolved_config['survey_type_detected'] = survey_type
        logger.info("Configuration resolved successfully.")

        # --- STAGE 1: DATA INGESTION AND CLEANING ---
        logger.info("--- STAGE 1: DATA INGESTION ---")

        # Initialize data ingestion components (Simplified)
        # (This assumes you've updated NSSCSVParser's __init__ to accept the config dict)
        parser = NSSCSVParser(resolved_config)
        cleaner = NSSDataCleaner(resolved_config)
        merger = NSSDataMerger(resolved_config)

        # Get file patterns from the resolved config
        file_types = resolved_config.get('file_types', {})
        if 'household' not in file_types or 'person' not in file_types:
            logger.error("Config missing 'household' or 'person' file_types.")
            return

        # Find and process household file
        household_patterns = file_types['household']['file_patterns']
        household_files = [f for pattern in household_patterns for f in Path(input_dir).glob(pattern)]
        if not household_files:
            raise FileNotFoundError(f"No household files found in {input_dir} with patterns: {household_patterns}")
        household_file = str(household_files[0])
        logger.info(f"Processing household file: {household_file}")

        household_df = parser.read_csv_file(household_file, 'household')
        household_df = cleaner.clean_dataframe(household_df, 'household')

        # Find and process person file
        person_patterns = file_types['person']['file_patterns']
        person_files = [f for pattern in person_patterns for f in Path(input_dir).glob(pattern)]
        if not person_files:
            raise FileNotFoundError(f"No person files found in {input_dir} with patterns: {person_patterns}")
        person_file = str(person_files[0])
        logger.info(f"Processing person file: {person_file}")

        person_df = parser.read_csv_file(person_file, 'person')
        person_df = cleaner.clean_dataframe(person_df, 'person')

        # Merge data
        logger.info("Merging household and person data...")
        merged_df = merger.merge_household_person_data(household_df, person_df)
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
        anonymized_df = privacy_enhancer.anonymize(analysis_df.copy())
        logger.info("Privacy enhancement complete.")

        # --- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---
        logger.info("--- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---")
        utility_assessor = NSSUtilityAssessor(resolved_config)
        utility_report = utility_assessor.run_utility_analysis(
            analysis_df,          # The original, clean data
            anonymized_df,        # The newly anonymized data
            risk_report.get('detected_quasi_identifiers', [])  # QIs from Module 1
        )
        logger.info(f"Utility Assessment Report: \n{json.dumps(utility_report, indent=2, default=str)}")

        # --- STAGE 5: SAVING OUTPUTS ---
        logger.info("--- STAGE 5: SAVING OUTPUTS ---")
        create_output_directory(output_dir)

        # Save the anonymized data
        output_format = resolved_config.get('output_format', 'parquet')
        output_filename = f'anonymized_{survey_type.lower()}_data.{output_format}'
        output_path = os.path.join(output_dir, output_filename)

        if output_format == 'parquet':
            anonymized_df.to_parquet(output_path, index=False)
        else:
            anonymized_df.to_csv(output_path, index=False)

        logger.info(f"Anonymized dataset saved to: {output_path}")

        # Save the risk report
        report_filename = f'risk_report_{survey_type.lower()}.json'
        report_path = os.path.join(output_dir, report_filename)
        with open(report_path, 'w') as f:
            json.dump(risk_report, f, indent=4)
        logger.info(f"Risk report saved to: {report_path}")

        # Save the utility report
        utility_report_filename = f'utility_report_{survey_type.lower()}.json'
        utility_report_path = os.path.join(output_dir, utility_report_filename)
        with open(utility_report_path, 'w') as f:
            json.dump(utility_report, f, indent=4, default=str)
        logger.info(f"Utility report saved to: {utility_report_path}")

        logger.info("==================================================")
        logger.info(" === NSS SAFEDATA PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("==================================================")

    except FileNotFoundError as fnf:
        logger.error(f"File Error: {fnf}")
    except Exception as e:
        logger.error(f"Pipeline failed with an unexpected error: {e}", exc_info=True)
        raise


def main():
    """Command line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='NSS SafeData Pipeline (Ingestion, Risk Assessment, Anonymization)'
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing NSS CSV files')
    parser.add_Fargument('--output-dir', required=True, help='Directory to save processed and anonymized data')
    parser.add_argument('--config', default='configs/ingestion_config.yaml', help='Path to the master configuration file')
    parser.add_argument('--survey-type', choices=['PLFS', 'HCES', 'ASI', 'EUS'],
                        help='Override survey type (auto-detect if not provided)')

    args = parser.parse_args()
    run_full_pipeline(args.input_dir, args.output_dir, args.config, args.survey_type)


if __name__ == '__main__':
    main()