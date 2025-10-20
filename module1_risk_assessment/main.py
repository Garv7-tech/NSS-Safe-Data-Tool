"""
Main entry point for NSS data ingestion pipeline. 
NOW WITH DYNAMIC SURVEY DETECTION
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import yaml
from data_ingestion.file_parser import NSSCSVParser
from data_ingestion.data_cleaner import NSSDataCleaner
from data_ingestion.data_merger import NSSDataMerger
from data_ingestion.survey_detector import NSSConfigResolver, NSSSurveyDetector
from data_ingestion.utils import setup_logging, get_memory_usage, create_output_directory

def run_ingestion_pipeline(input_dir : str, output_dir : str, config_path : str, survey_path : str = None):

    """
    Run the complete NSS data ingestion pipeline
    
    Args:
        input_dir: Directory containing NSS CSV files
        output_dir: Directory to save processed data
        config_path: Path to configuration file
        survey_type: Optional survey type override (auto-detect if None)
    """

    logger = setup_logging(__name__)
    logger.info("Starting NSS data ingestion pipeline with dynamic survey detection")

    try : 

        # Load full multi-survey configuration
        full_config = load_full_configuration(config_path)

        # AUTO DETECT SURVEY TYPE IF NOT PROVIDED
        if not survey_type:
            detector = NSSSurveyDetector(full_config)
            survey_type = detector.detect_survey_type(input_dir)

        logger.info(f"Using survey type : {survey_type}")

        #  RESOLVE CONFIG FOR DETECTED SURVEY TYPE
        resolver = NSSConfigResolver(full_config)
        resolved_config = resolver.resolve_config_for_survey(survey_type)

        # Initialize components
        parser = NSSCSVParser.__new__(NSSCSVParser) # Create without calling __init__

        parser.config = resolved_config # Inject resolved config directly
        parser.logger = setup_logging(parser.__class__.__name__)

        cleaner = NSSDataCleaner(parser.config)
        merger = NSSDataMerger(parser.config)

        # Cleaner output directory
        create_output_directory(output_dir)

        # Get file patterns for this survey type
        file_types = resolved_config['file_types']


        # Find and process household file
        household_patterns = file_types['household']['file_patterns']
        household_files = []

        for pattern in household_patterns:
            household_files.extend(list(Path(input_dir).glob(pattern)))

        if not household_files:
            raise FileNotFoundError(f"No household files found with patterns: {household_patterns}")
        
        household_file = str(household_files[0])
        logger.info(f"Processing household file : {household_file}")

        # Read and clean household data
        household_df = parser.read_csv_file(household_file, 'household')
        household_df = cleaner.clean_dataframe(household_df, 'household')

        # Find and process person file
        person_patterns = file_types['person']['file_patterns']
        person_files = []

        for pattern in person_patterns:
            person_files.extend(list(Path(input_dir).glob(pattern)))

        if not person_files:
            raise FileNotFoundError(f"No person files found with patterns: {person_patterns}")
        
        person_file = str(person_files[0])
        logger.info(f"Processing person file: {person_file}")

        # Read and clean person data
        person_df = parser.read_csv_file(person_file, 'person')
        person_df = cleaner.clean_dataframe(person_df, 'person')

        # Merge data
        logger.info("Merging household and person data")
        merged_df = merger.merge_household_person_data(household_df, person_df)

        # Create analysis dataset
        analysis_df = merger.create_analysis_ready_dataset(merged_df)

        # Save processed data
        output_filename = f'nss_{survey_type.lower()}_analysis_data.parquet'
        output_path = os.path.join(output_dir, output_filename)
        analysis_df.to_parquet(output_path, index=False)
        logger.info(f"Analysis dataset saved to: {output_path}")

        # Log memory usage and statistics
        memory_stats = get_memory_usage(analysis_df)
        logger.info(f"Final dataset shape: {memory_stats['shape']}")
        logger.info(f"Memory usage: {memory_stats['total_memory_mb']:.2f} MB")
        logger.info(f"Survey type processed: {survey_type}")
        
        logger.info("NSS data ingestion pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed : {e}")
        raise



def main():
    """Command line interface with optional survey type override"""
    parser = argparse.ArgumentParser(description='NSS Data Ingestion Pipeline - Multi-Survey Support')
    parser.add_argument('--input-dir', required=True, help='Directory containing NSS CSV files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed data')
    parser.add_argument('--config', default='configs/ingestion_config.yaml', help='Path to configuration file')
    parser.add_argument('--survey-type', choices=['PLFS', 'HCES', 'ASI', 'EUS'],help='Override survey type (auto-detect if not provided)')
    
    args = parser.parse_args()
    
    run_ingestion_pipeline(args.input_dir, args.output_dir, args.config, args.survey_type)













#     # Auto-detect survey type (works with any NSS survey)
# python main.py --input-dir ./data/plfs_files --output-dir ./output

# # Override survey type explicitly  
# python main.py --input-dir ./data/hces_files --output-dir ./output --survey-type HCES

# # Works with ASI, EUS, or any other configured survey
# python main.py --input-dir ./data/asi_files --output-dir ./output
 