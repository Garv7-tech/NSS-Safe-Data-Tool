"""
Tests for Module 4: Report Generator
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock, ANY

# Fixtures are loaded from the root conftest.py
from conftest import test_config

# --- Mock Data Fixtures ---

@pytest.fixture
def mock_risk_report():
    """Mock output from Module 1"""
    return {
        'detected_quasi_identifiers': ['Age', 'Sex', 'District_Code'],
        'risk_metrics': {
            'k_anonymity': 1,
            'total_records': 100
        },
        'linkage_attack_simulation': {
            'vulnerable_records_k1': 40,
            'reidentification_rate_percent': 40.0
        }
    }

@pytest.fixture
def mock_privacy_report():
    """Mock output from Module 2"""
    return {
        'anonymization_strategy': 'k_anonymity',
        'goals_met': True,
        'final_k': 10,
        'final_info_loss_ncp': 0.15,
        'quasi_identifiers_processed': ['Age', 'Sex', 'District_Code']
    }

@pytest.fixture
def mock_utility_report():
    """Mock output from Module 3"""
    return {
        'statistical_comparison': {
            'Age': {'original_mean': 38.0, 'anonymized_mean': 37.5}
        },
        'correlation_comparison': {
            'mean_absolute_correlation_difference': 0.05
        },
        'ml_model_comparison': {
            'task_type': 'regression',
            'original_score (r2_score)': 0.90,
            'anonymized_score (r2_score)': 0.85,
            'performance_drop_percent': 5.55
        },
        'anonymized_data_risk': {
            'final_k_anonymity': 10,
            'vulnerable_records_k1': 0,
            'reidentification_rate_percent': 0.0
        }
    }

# --- Tests ---

def test_generate_json_report(test_config, mock_risk_report, mock_privacy_report, mock_utility_report):
    """
    Tests that the JSON report is structured correctly and aggregates data.
    """
    # We must import the class *after* the reportlab mock (if used),
    # but for this test, it's fine.
    from module4_reporting.report_generator import NSSReportGenerator
    
    generator = NSSReportGenerator(test_config)
    
    json_report = generator.generate_json_report(
        mock_risk_report,
        mock_privacy_report,
        mock_utility_report
    )
    
    # Test high-level structure
    assert 'summary' in json_report
    assert 'initial_risk_assessment' in json_report
    assert 'privacy_enhancement_details' in json_report
    assert 'utility_assessment' in json_report
    
    # Test summary data aggregation
    summary = json_report['summary']
    assert summary['pipeline_status'] == "Success"
    assert summary['survey_type'] == "PLFS"
    assert summary['final_k_anonymity'] == 10
    assert summary['vulnerable_records_remaining'] == 0
    assert summary['final_re_id_risk_percent'] == 0.0
    assert summary['ml_utility_score'] == 0.85
    
    # Test detail sections
    assert json_report['initial_risk_assessment']['risk_metrics']['k_anonymity'] == 1
    assert json_report['privacy_enhancement_details']['anonymization_strategy'] == 'k_anonymity'
    assert json_report['utility_assessment']['correlation_comparison']['mean_absolute_correlation_difference'] == 0.05

@patch('module4_reporting.report_generator.SimpleDocTemplate')
@patch('module4_reporting.report_generator.Paragraph')
@patch('module4_reporting.report_generator.Spacer')
@patch('module4_reporting.report_generator.Preformatted')
def test_save_pdf_report(mock_preformatted, mock_spacer, mock_paragraph, mock_doc_template, 
                        test_config, mock_risk_report, mock_privacy_report, mock_utility_report):
    """
    Tests that the PDF generation is triggered correctly, without writing a file.
    We mock the ReportLab classes and check if 'build' method was called.
    """
    # Need to import the class *after* the mocks are in place
    from module4_reporting.report_generator import NSSReportGenerator

    # Create a mock instance for the document template
    mock_doc_instance = MagicMock()
    mock_doc_template.return_value = mock_doc_instance
    
    generator = NSSReportGenerator(test_config)
    
    json_report = generator.generate_json_report(
        mock_risk_report,
        mock_privacy_report,
        mock_utility_report
    )
    
    output_path = "dummy/path/report.pdf"
    generator.save_pdf_report(json_report, output_path)
    
    # Check that SimpleDocTemplate was initialized with the correct path
    mock_doc_template.assert_called_with(output_path, pagesize=ANY, 
                                        rightMargin=ANY, leftMargin=ANY, 
                                        topMargin=ANY, bottomMargin=ANY)
    
    # Check that the 'build' method was called on the instance
    # This confirms the story was built and the PDF was (in theory) saved.
    mock_doc_instance.build.assert_called_once()
    
    # Check that our main sections were added (Paragraphs were created)
    assert any("Executive Summary" in call[0][0] for call in mock_paragraph.call_args_list)
    assert any("Initial Risk Assessment" in call[0][0] for call in mock_paragraph.call_args_list)
    assert any("Utility Assessment" in call[0][0] for call in mock_paragraph.call_args_list)
    
    # Check that the JSON content was added (Preformatted was created)
    assert mock_preformatted.call_count >= 3