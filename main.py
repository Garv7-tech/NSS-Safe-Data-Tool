"""
Main entry point for the NSS SafeData Pipeline.
This script orchestrates the full pipeline:
0. Config Auto-Generation (NEW)
1. Data Ingestion (Parsing, Cleaning, Merging)
2. Risk Assessment (Module 1: QI Detection & Linkage Attack)
3. Privacy Enhancement (Module 2: Anonymization)
4. Utility Assessment (Module 3: Utility Measurement)
5. Report Generation (Module 4: Reporting)
"""

import argparse
import logging
import pandas as pd
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# --- NEW: Module 0 Imports ---
# (Create this new folder and files from my previous response)
try:
    from module0_config_generation.schema_detector import SchemaDetector
    from module0_config_generation.config_writer import ConfigWriter
except ImportError:
    print("Warning: module0_config_generation not found. Auto-config will be skipped.")
    SchemaDetector = None
    ConfigWriter = None

# --- Module 1 Imports ---
from module1_risk_assessment.data_ingestion.survey_detector import SurveyDetector
from module1_risk_assessment.data_ingestion.file_parser import FileParser
from module1_risk_assessment.data_ingestion.data_cleaner import DataCleaner
from module1_risk_assessment.data_ingestion.data_merger import DataMerger
from module1_risk_assessment.risk_assessor import RiskAssessor

# --- Module 2 Imports ---
from module2_privacy_enhancement.privacy_enhancer import PrivacyEnhancer

# --- Module 3 Imports ---
from module3_utility_assessment.utility_assessor import UtilityAssessor

# --- Module 4 Imports ---
from module4_reporting.report_generator import ReportGenerator
from module4_reporting.config_models import load_pipeline_config, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- NEW STAGE 0 ---
def run_stage_0_auto_config(
    input_dir: Path, 
    config_path: Path, 
    parser: FileParser,
    detector: SurveyDetector
) -> bool:
    """
    NEW STAGE 0: Auto-detects schema for unknown surveys and updates config.
    Returns True if config was updated, False otherwise.
    """
    logger.info("--- STAGE 0: CONFIGURATION AUTO-DETECTION ---")
    
    if ConfigWriter is None or SchemaDetector is None:
        logger.warning("module0_config_generation not found. Skipping auto-config.")
        return False
        
    config_updated = False
    
    # Check all files in the input directory
    all_files = list(input_dir.glob('*.csv')) + list(input_dir.glob('*.dta'))
    if not all_files:
        logger.warning(f"No data files found in {input_dir}. Skipping auto-config.")
        return False
        
    writer = ConfigWriter(config_path)
    writer.load_config() # Load the existing config
    
    for file_path in all_files:
        # Check if this survey is *already known*
        survey_type, _ = detector.detect_survey(file_path.name, file_path)
        
        # If it's 'default_survey', it means it was not recognized
        if survey_type == 'default_survey':
            logger.warning(f"Found unknown survey file: {file_path.name}. Attempting auto-detection...")
            
            try:
                # 1. Read the file
                df = parser.parse_file(file_path)
                
                # 2. Detect its schema
                # We'll use the file name as the survey name
                new_survey_name = file_path.stem.split('_')[0] 
                if not new_survey_name or new_survey_name == '*':
                    new_survey_name = 'new_survey'
                    
                schema_detector = SchemaDetector(df, new_survey_name)
                new_schema = schema_detector.infer_schema()
                
                # 3. Write new schema to config file
                writer.add_new_survey(
                    survey_name=new_survey_name,
                    schema=new_schema,
                    file_pattern=f"*{file_path.stem}*" # Add a pattern to recognize it next time
                )
                config_updated = True
                
            except Exception as e:
                logger.error(f"Failed to auto-detect schema for {file_path.name}: {e}", exc_info=True)
                
    if config_updated:
        writer.write_config()
        logger.info("ingestion_config.yaml has been updated with new surveys.")
    else:
        logger.info("No new surveys found. Config file is unchanged.")
        
    return config_updated


# --- STAGE 1 ---
def run_stage_1_data_ingestion(
    input_dir: Path, 
    config: PipelineConfig,
    detector: SurveyDetector
) -> Optional[pd.DataFrame]:
    """STAGE 1: Detects, parses, cleans, and merges survey data."""
    logger.info("--- STAGE 1: DATA INGESTION ---")
    
    try:
        # 1. Detect Survey
        # For simplicity, we'll assume one survey type per directory
        # A more complex setup would group files by detected type
        all_files = list(input_dir.glob('*.csv')) + list(input_dir.glob('*.dta'))
        if not all_files:
            logger.error("No data files found in input directory.")
            return None
            
        first_file = all_files[0]
        survey_type, survey_config_dict = detector.detect_survey(first_file.name, first_file)
        
        if survey_type == 'default_survey' and config.pipeline.default_survey:
            survey_type = config.pipeline.default_survey
            survey_config_dict = config.surveys.get(survey_type)
            logger.info(f"Using default survey config: {survey_type}")
        
        if not survey_config_dict:
            logger.error(f"Could not find valid config for files in: {input_dir}")
            return None

        logger.info(f"Processing as survey type: {survey_type}")
        
        # 2. Parse, Clean, Merge (using components)
        parser = FileParser()
        cleaner = DataCleaner(survey_config_dict)
        merger = DataMerger(survey_config_dict)
        
        # This part needs to be robust. Assuming simple structure for now.
        # A real implementation would loop through file types in config
        
        # Let's use the DataMerger's logic if available
        if hasattr(merger, 'merge_data'):
             # Assuming merge_data handles parsing and cleaning
            analysis_df = merger.merge_data(input_dir, parser, cleaner)
        else:
            # Fallback to simple parse/clean of the first file
            logger.warning("Using simplified ingestion (first file only).")
            df = parser.parse_file(first_file)
            analysis_df = cleaner.clean_data(df)
            
        if analysis_df is None or analysis_df.empty:
            logger.error("Data ingestion resulted in an empty DataFrame.")
            return None

        logger.info(f"Data ingestion complete. Shape: {analysis_df.shape}")
        return analysis_df, survey_config_dict
    
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        return None


# --- STAGE 2 ---
def run_stage_2_risk_assessment(
    analysis_df: pd.DataFrame, 
    survey_config: Dict[str, Any],
    ground_truth_file: Optional[Path] # --- NEW ARGUMENT ---
) -> Dict[str, Any]:
    """STAGE 2: Assesses k-anonymity risk and runs linkage attack."""
    # --- UPDATED CALL ---
    assessor = RiskAssessor(analysis_df, survey_config)
    risk_report = assessor.assess_risk(ground_truth_file=ground_truth_file)
    return risk_report


# --- STAGE 3 ---
def run_stage_3_privacy_enhancement(
    analysis_df: pd.DataFrame, 
    survey_config: Dict[str, Any], 
    pipeline_config: PipelineConfig
) -> pd.DataFrame:
    """STAGE 3: Applies privacy techniques (k-anonymity, SDG)."""
    logger.info("--- STAGE 3: PRIVACY ENHANCEMENT (MODULE 2) ---")
    enhancer = PrivacyEnhancer(analysis_df, survey_config, pipeline_config.privacy_enhancement)
    anonymized_df = enhancer.enhance_privacy()
    logger.info("Privacy enhancement complete.")
    return anonymized_df


# --- STAGE 4 ---
def run_stage_4_utility_assessment(
    original_df: pd.DataFrame, 
    anonymized_df: pd.DataFrame, 
    survey_config: Dict[str, Any]
) -> Dict[str, Any]:
    """STAGE 4: Compares utility of original vs. anonymized data."""
    # --- UPDATED CALL ---
    # The new UtilityAssessor will be called, which includes distribution checks.
    assessor = UtilityAssessor(original_df, anonymized_df, survey_config)
    utility_report = assessor.assess_utility()
    return utility_report


# --- STAGE 5 ---
def run_stage_5_reporting(
    risk_report: Dict[str, Any], 
    utility_report: Dict[str, Any], 
    output_dir: Path
):
    """STAGE 5: Generates the final PDF and JSON reports."""
    logger.info("--- STAGE 5: REPORTING (MODULE 4) ---")
    reporter = ReportGenerator(risk_report, utility_report, output_dir)
    reporter.generate_json_report()
    reporter.generate_pdf_report()
    logger.info(f"Reports saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="NSS SafeData Pipeline")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        required=True, 
        help="Directory containing input survey data files (CSV, DTA)."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True, 
        help="Directory to save anonymized data and reports."
    )
    parser.add_argument(
        "--config-file", 
        type=Path, 
        default=Path(__file__).parent / "module1_risk_assessment/configs/ingestion_config.yaml",
        help="Path to the main ingestion_config.yaml file."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=None,
        help="Optional path to a 'ground truth' CSV file for linkage attack simulation."
    )
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- (Existing Stage 0 is renamed to 'Config Loading') ---
        logger.info("--- PIPELINE STARTING ---")
        logger.info("--- CONFIG LOADING ---")
        config = load_pipeline_config(args.config_file)
        
        # Initialize common components
        file_parser = FileParser()
        survey_detector = SurveyDetector(config)

        # --- RUN NEW STAGE 0 ---
        config_was_updated = run_stage_0_auto_config(
            args.input_dir, 
            args.config_file, 
            file_parser,
            survey_detector
        )
        
        # If config was updated, we must reload it to continue
        if config_was_updated:
            logger.info("Reloading configuration after auto-detection...")
            config = load_pipeline_config(args.config_file)
            survey_detector = SurveyDetector(config) # Re-init detector with new rules


        # --- RUN STAGE 1: INGESTION ---
        ingestion_result = run_stage_1_data_ingestion(args.input_dir, config, survey_detector)
        if ingestion_result is None:
            raise RuntimeError("Data Ingestion failed. See logs for details.")
        analysis_df, survey_config = ingestion_result
        
        # Save original (cleaned, merged) data for reference
        survey_name = survey_config.get('name', 'survey')
        original_data_path = args.output_dir / f"original_{survey_name}.parquet"
        analysis_df.to_parquet(original_data_path)
        logger.info(f"Saved original cleaned data to {original_data_path}")

        # --- RUN STAGE 2: RISK ASSESSMENT ---
        risk_report = run_stage_2_risk_assessment(
            analysis_df, 
            survey_config,
            args.ground_truth_file # Pass the new argument
        )

        # --- RUN STAGE 3: PRIVACY ENHANCEMENT ---
        anonymized_df = run_stage_3_privacy_enhancement(
            analysis_df, 
            survey_config, 
            config
        )

        # Save anonymized data
        anonymized_data_path = args.output_dir / f"anonymized_{survey_name}.parquet"
        anonymized_df.to_parquet(anonymized_data_path)
        logger.info(f"Saved anonymized data to {anonymized_data_path}")

        # --- RUN STAGE 4: UTILITY ASSESSMENT ---
        utility_report = run_stage_4_utility_assessment(
            analysis_df, 
            anonymized_df, 
            survey_config
        )

        # --- RUN STAGE 5: REPORTING ---
        run_stage_5_reporting(
            risk_report, 
            utility_report, 
            args.output_dir
        )

        logger.info("--- PIPELINE COMPLETED SUCCESSFULLY ---")

    except Exception as e:
        logger.error(f"--- PIPELINE FAILED ---", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()