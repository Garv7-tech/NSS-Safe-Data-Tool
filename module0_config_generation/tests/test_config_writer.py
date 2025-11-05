import pytest
import yaml
from pathlib import Path
from module0_config_generation.config_writer import ConfigWriter
from unittest.mock import mock_open, patch

@pytest.fixture
def mock_config_data(mocker):
    """Mocks the initial config.yaml data."""
    return {
        'encoding': 'utf-8',
        'surveys': {
            'PLFS': {'survey_name': 'Periodic Labour Force Survey'}
        },
        'survey_detection': {
            'file_patterns': {'PLFS': ['*hhv*.csv']},
            'column_signatures': {'PLFS': ['Panel', 'Age']}
        },
        'risk_assessment': {
            'quasi_identifier_candidates': {
                'common': ['Age'],
                'survey_specific': {'PLFS': ['Education']}
            }
        },
        'analysis_columns': {
            'core_identifiers': {'PLFS': ['HHID']},
            'demographics': {'PLFS': ['Age', 'Sex']}
        }
    }

@pytest.fixture
def new_survey_data():
    """Sample output from SchemaDetector for a new 'HCES' survey."""
    survey_config = {
        'survey_name': 'Auto-Generated HCES Survey',
        'file_types': {'primary_file': {'column_mapping': {'h_age': 'Age'}}}
    }
    detection_rules = {
        'file_patterns': {'HCES': ['*hces*']},
        'column_signatures': {'HCES': ['h_age', 'dist']}
    }
    master_additions = {
        'risk_assessment': {'survey_specific': {'HCES': ['h_age', 'dist']}},
        'analysis_columns': {
            'core_identifiers': {'HCES': ['hh_id']},
            'demographics': {'HCES': ['h_age']}
        }
    }
    return 'HCES', survey_config, detection_rules, master_additions

def test_config_writer_adds_new_survey(mocker, mock_config_data, new_survey_data):
    mock_file_path = Path('/fake/config.yaml')
    
    # Mock file operations
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch('os.replace')
    
    # Mock open() for reading the old config and writing the new one
    m = mock_open(read_data=yaml.dump(mock_config_data))
    mocker.patch('builtins.open', m)

    writer = ConfigWriter(mock_file_path)
    
    # Run the function
    writer.add_new_survey(*new_survey_data)

    # Check that the config data in memory is updated correctly
    assert 'PLFS' in writer.config_data['surveys']
    assert 'HCES' in writer.config_data['surveys']
    assert writer.config_data['surveys']['HCES']['survey_name'] == 'Auto-Generated HCES Survey'
    
    assert 'HCES' in writer.config_data['survey_detection']['file_patterns']
    assert 'HCES' in writer.config_data['risk_assessment']['quasi_identifier_candidates']['survey_specific']
    assert 'h_age' in writer.config_data['risk_assessment']['quasi_identifier_candidates']['survey_specific']['HCES']
    assert 'HCES' in writer.config_data['analysis_columns']['demographics']

    # Check that write was called
    m.assert_called_with(mock_file_path, 'w')
    handle = m()
    handle.write.assert_called()