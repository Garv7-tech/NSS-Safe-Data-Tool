"""
Module 4: Pydantic Configuration Models
Provides robust, nested data models for parsing and validating the 
entire 'ingestion_config.yaml' file.

This acts as the "single source of truth" for all pipeline configurations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any

class FileTypeConfig(BaseModel):
    """Configuration for a specific file type (e.g., household, person)"""
    file_patterns: List[str]
    required_columns: List[str]
    identifier_columns: List[str]
    dtypes: Dict[str, str] = {}
    date_columns: List[str] = []
    missing_value_rules: Dict[str, str] = {}
    range_validation: Dict[str, Dict[str, float]] = {}
    categorical_mappings: Dict[str, Dict[int, str]] = {}
    column_mapping: Optional[Dict[str, str]] = None # For standardizing names

class SurveyConfig(BaseModel):
    """Configuration specific to a single survey (e.g., PLFS, HCES)"""
    survey_name: str
    file_types: Dict[str, FileTypeConfig]

class SurveyDetectionConfig(BaseModel):
    """Rules for auto-detecting the survey type"""
    file_patterns: Dict[str, List[str]]
    column_signatures: Dict[str, List[str]]

class MergeStrategiesConfig(BaseModel):
    """Default and survey-specific merge keys"""
    default_keys: List[str]
    survey_specific: Dict[str, List[str]] = {}

class AnalysisColumnsConfig(BaseModel):
    """Columns to select for the final analysis dataset, by category"""
    core_identifiers: Dict[str, List[str]]
    demographics: Dict[str, Any] # Can be common or survey_specific
    household_characteristics: Dict[str, Any]
    employment: Dict[str, List[str]] = {}
    expenditure: Dict[str, List[str]] = {}

class RiskAssessmentConfig(BaseModel):
    """Configuration for Module 1: Risk Assessment"""
    quasi_identifier_candidates: Dict[str, Any]

class KAnonymityGoals(BaseModel):
    """Specific goals for the k-anonymity strategy"""
    target_k: int = Field(gt=1)
    max_info_loss: float = Field(ge=0.0, le=1.0)

class DifferentialPrivacyGoals(BaseModel):
    """Specific goals for the differential privacy strategy"""
    epsilon: float = Field(gt=0.0)

class PrivacyGoalsConfig(BaseModel):
    """Nested goals for different strategies"""
    k_anonymity: Optional[KAnonymityGoals] = None
    differential_privacy: Optional[DifferentialPrivacyGoals] = None

class PrivacyEnhancementConfig(BaseModel):
    """Configuration for Module 2: Privacy Enhancement"""
    privacy_strategy: str
    goals: PrivacyGoalsConfig

class MLModelConfig(BaseModel):
    """Configuration for the ML utility test in Module 3"""
    task_type: str
    target_column: str
    feature_columns: List[str]

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        if v not in ['regression', 'classification']:
            raise ValueError("task_type must be 'regression' or 'classification'")
        return v

class UtilityAssessmentConfig(BaseModel):
    """Configuration for Module 3: Utility Assessment"""
    numeric_columns_to_compare: List[str]
    ml_model: MLModelConfig

class FileProcessingConfig(BaseModel):
    """Global file processing settings"""
    max_file_size_mb: int
    supported_formats: List[str]
    auto_detect_delimiter: bool
    delimiter_candidates: List[str]

class QualityChecksConfig(BaseModel):
    """Global data quality check settings"""
    duplicate_detection: Dict[str, Any]
    missing_data_threshold: float
    outlier_detection: Dict[str, Any]
    referential_integrity: Dict[str, Any]

class OutputConfig(BaseModel):
    """Global output settings"""
    save_intermediate: bool
    intermediate_formats: List[str]
    final_formats: List[str]
    include_metadata: bool
    create_data_dictionary: bool

class MasterConfig(BaseModel):
    """
    The top-level model for the entire ingestion_config.yaml file.
    This is the "single source of truth".
    """
    encoding: str
    na_values: List[str]
    chunk_size: int
    output_format: str
    default_survey: str
    
    surveys: Dict[str, SurveyConfig]
    merge_strategies: Dict[str, MergeStrategiesConfig]
    analysis_columns: AnalysisColumnsConfig
    survey_detection: SurveyDetectionConfig
    file_processing: FileProcessingConfig
    quality_checks: QualityChecksConfig
    output: OutputConfig
    risk_assessment: RiskAssessmentConfig
    privacy_enhancement: PrivacyEnhancementConfig
    utility_assessment: UtilityAssessmentConfig

    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v):
        if v not in ['parquet', 'csv']:
            raise ValueError("output_format must be 'parquet' or 'csv'")
        return v