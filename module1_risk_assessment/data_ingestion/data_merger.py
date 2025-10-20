"""
NSS Data Merger
Handles merging of household and person-level data
"""

import pandas as pd
from typing import Dict, List, Optional
import logging
from .utils import setup_logging


class NSSDataMerger:
    """
    Merger for NSS household and person-level data
    """

    def __init__(self, config:Dict):
        """
        Initialize merger with configuration
        
        Args:
            config: Configuration dictionary
        """

        self.config = config
        self.logger = setup_logging(__name__)

    def merge_household_person_data(self, household_df : pd.DataFrame, person_df : pd.DataFrame)-> pd.DataFrame:
        """
        Merge household and person-level data
        
        Args:
            household_df: Household-level DataFrame
            person_df: Person-level DataFrame
            
        Returns:
            Merged DataFrame
        """

        # Get merge keys from configuration

        merge_keys = self.config.get('merge_keys',{
            'household_person' : ['Panel', 'Sample_HouseHold_Number', 'State_Ut_code', 'District_Code']
        })

        household_keys = merge_keys['household_person']

        try:
            # Validate merge keys exist in both dataframes
            missing_keys_hh = set(household_keys) - set(household_df.columns)
            missing_keys_person = set(household_keys)- set(person_df.columns)

            if missing_keys_hh:
                raise ValueError(f"Missing keys in household data: {missing_keys_hh}")
            if missing_keys_person:
                raise ValueError(f"Missing keys in person data: {missing_keys_person}")
            
            # Perform merge

            merged_df = person_df.merge(
                household_df,
                on = household_keys,
                how = 'left',
                suffixes = ('_person', '_household')
            )

            self.logger.info(f"Successfully merged data: {len(merged_df)} records")
            self.logger.info(f"Original person records: {len(person_df)}")
            self.logger.info(f"Original household records: {len(household_df)}")


            # Validate merge success
            unmatched_records = merged_df.isnull().any(axis = 1).sum()

            if unmatched_records > 0:
                self.logger.warning(f"{unmatched_records} person records without household match")

            return merged_df
        
        except Exception as e:
            self.logger.error(f"Merge failed : {e}")
            raise


    def create_analysis_ready_dataset(self, merged_df : pd.DataFrame)-> pd.DataFrame:
        """
        Create final analysis-ready dataset with selected columns
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Analysis-ready DataFrame
        """

        # Get important columns from configuration
        important_columns = self.config.get('analysis_columns', {
            'identifiers' : ['Panel', 'Sample_Household_Number', 'Person_Serial_No'],
            'demographics' : ['Age', 'Sex', 'Marital_Status', 'Social_Group'],
            'household' : ['Household_Size', 'Household_Type', 'Monthly_Consumer_Expenditure'],
            'employment' : ['Principal_Status_Code', 'Principal_Industry_Code']
        })

        # Flatten column lists
        all_columns = []
        for category, columns in important_columns.items():
            all_columns.extend(columns)

        # Select available columns
        available_columns = [col for col in all_columns if col in merged_df.columns]

        analysis_df = merged_df[available_columns].copy()

        self.logger.info(f"Analysis dataset created with {len(available_columns)}")

        return analysis_df