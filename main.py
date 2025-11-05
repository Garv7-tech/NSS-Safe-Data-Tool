"""
Main entry point for the NSS SafeData Pipeline.
This script orchestrates the full pipeline:
1. Data Ingestion (Parsing, Cleaning, Merging)
2. Risk Assessment (Module 1: QI Detection & Risk Scoring)
3. Privacy Enhancement (Module 2: Anonymization)
4. Utility Assessment (Module 3: Utility Measurement)
5. Report Generation (Module 4: Reporting)
"""

import argparse
import os
import json
import yaml
from pathlib import Path
import pandas as pd
import logging
from pydantic import ValidationError

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

# --- Imports from Module 4 ---
from module4_reporting.config_models import MasterConfig
from module4_reporting.report_generator import NSSReportGenerator


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
            
            # [NEW] Validate the entire configuration using Pydantic
            master_config = MasterConfig(**full_config)
            logger.info("Master configuration file successfully loaded and validated.")
        
        except ValidationError as e:
            logger.error(f"Configuration validation failed for {config_path}: \n{e}")
            return
        except Exception as e:
            logger.error(f"Failed to load configuration file from {config_path}: {e}")
            return

        # Auto-detect survey type if not provided
        if not survey_type:
            logger.info("No survey type provided, attempting auto-detection...")
            detector = NSSSurveyDetector(master_config.model_dump())
            survey_type = detector.detect_survey_type(input_dir)
        logger.info(f"Using Survey Type: {survey_type}")

        # Resolve the configuration for the detected survey
        resolver = NSSConfigResolver(master_config.model_dump())
        resolved_config = resolver.resolve_config_for_survey(survey_type)
        resolved_config['survey_type_detected'] = survey_type
        logger.info("Configuration resolved for survey.")

        # --- STAGE 1: DATA INGESTION AND CLEANING ---
        logger.info("--- STAGE 1: DATA INGESTION ---")

        # Initialize data ingestion components
        # Note: This assumes constructors for Parser/Cleaner/Merger
        # are updated to take the resolved_config dictionary.
        parser = NSSCSVParser(resolved_config) 
        cleaner = NSSDataCleaner(resolved_config)
        merger = NSSDataMerger(resolved_config)
        
        file_types = resolved_config.get('file_types', {})
        household_patterns = file_types.get('household', {}).get('file_patterns', [])
        household_files = [f for pattern in household_patterns for f in Path(input_dir).glob(pattern)]
        if not household_files:
            raise FileNotFoundError(f"No household files found in {input_dir} with patterns: {household_patterns}")
        household_df = parser.read_csv_file(str(household_files[0]), 'household')
        household_df = cleaner.clean_dataframe(household_df, 'household')

        person_patterns = file_types.get('person', {}).get('file_patterns', [])
        person_files = [f for pattern in person_patterns for f in Path(input_dir).glob(pattern)]
        if not person_files:
            raise FileNotFoundError(f"No person files found in {input_dir} with patterns: {person_patterns}")
        person_df = parser.read_csv_file(str(person_files[0]), 'person')
        person_df = cleaner.clean_dataframe(person_df, 'person')

        logger.info("Merging household and person data...")
        merged_df = merger.merge_household_person_data(household_df, person_df)
        analysis_df = merger.create_analysis_ready_dataset(merged_df)
        logger.info(f"Data ingestion complete. Analysis dataset shape: {analysis_df.shape}")

        # --- STAGE 2: RISK ASSESSMENT (MODULE 1) ---
        logger.info("--- STAGE 2: RISK ASSESSMENT (MODULE 1) ---")
        risk_assessor = NSSRiskAssessor(resolved_config)
        risk_report = risk_assessor.run_risk_analysis(analysis_df)
        # Add total record count to risk report for summary
        risk_report.get('risk_metrics', {})['total_records'] = len(analysis_df)
        logger.info("Risk assessment complete.")

        # --- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---
        logger.info("--- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---")
        privacy_enhancer = NSSPrivacyEnhancer(resolved_config, risk_report)
        anonymized_df = privacy_enhancer.anonymize(analysis_df.copy())
        
        # Create a simple privacy report for Module 4
        # (This can be enhanced in Module 2 to be more detailed)
        privacy_report = {
            "anonymization_strategy": resolved_config.get('privacy_enhancement', {}).get('privacy_strategy'),
            "quasi_identifiers_processed": risk_report.get('detected_quasi_identifiers'),
            "goals": resolved_config.get('privacy_enhancement', {}).get('goals'),
            "anonymization_status": "Success" if not analysis_df.equals(anonymized_df) else "Failed (Data returned unanonymized)"
        }
        logger.info("Privacy enhancement complete.")

        # --- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---
        logger.info("--- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---")
        utility_assessor = NSSUtilityAssessor(resolved_config)
        utility_report = utility_assessor.run_utility_analysis(
            analysis_df,          # The original, clean data
            anonymized_df,        # The newly anonymized data
            risk_report.get('detected_quasi_identifiers', [])  # QIs from Module 1
        )
        logger.info("Utility assessment complete.")

        # --- STAGE 5: GENERATE FINAL REPORTS (MODULE 4) ---
        logger.info("--- STAGE 5: GENERATE FINAL REPORTS (MODULE 4) ---")
        report_generator = NSSReportGenerator(resolved_config)
        
        # Generate the consolidated JSON data
        final_report_data = report_generator.generate_json_report(
            risk_report,
            privacy_report,
            utility_report
        )
        logger.info("Consolidated JSON report generated.")

        # --- STAGE 6: SAVING OUTPUTS ---
        logger.info("--- STAGE 6: SAVING OUTPUTS ---")
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

        # Save the consolidated JSON report (for the frontend)
        report_path_json = os.path.join(output_dir, 'final_pipeline_report.json')
        with open(report_path_json, 'w') as f:
            json.dump(final_report_data, f, indent=4, default=str)
        logger.info(f"Final JSON report saved to: {report_path_json}")

        # Save the consolidated PDF report (for download)
        report_path_pdf = os.path.join(output_dir, 'final_pipeline_report.pdf')
        report_generator.save_pdf_report(final_report_data, report_path_pdf)
        # Logger message for PDF is inside the generator class

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
        description='NSS SafeData Pipeline (Ingestion, Risk Assessment, Anonymization, Utility, Reporting)'
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing NSS CSV files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed data and reports')
    
    # Updated default config path to be more standard
    default_config_path = 'module1_risk_assessment/configs/ingestion_config.yaml'
    parser.add_argument('--config', default=default_config_path, help=f'Path to the master configuration file (default: {default_config_path})')
    
    parser.add_argument('--survey-type', choices=['PLFS', 'HCES', 'ASI', 'EUS'],
                        help='Override survey type (auto-detect if not provided)')

    args = parser.parse_args()
    
    config_file_path = args.config
    if not os.path.exists(config_file_path):
        # Try to find it relative to the script
        script_dir = os.path.dirname(__file__)
        alt_path = os.path.join(script_dir, config_file_path)
        if os.path.exists(alt_path):
            config_file_path = alt_path
        else:
            print(f"Error: Configuration file not found at {args.config} or {alt_path}")
            return

    run_full_pipeline(args.input_dir, args.output_dir, config_file_path, args.survey_type)


if __name__ == '__main__':
    main()