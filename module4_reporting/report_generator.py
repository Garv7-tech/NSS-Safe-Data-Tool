"""
Module 4: NSS Report Generator
Consolidates all analysis from Modules 1, 2, and 3 into comprehensive
JSON and PDF reports for the end-user.
"""

import json
import pandas as pd
from typing import Dict, Any, List
import logging
from module1_risk_assessment.data_ingestion.utils import setup_logging

# --- PDF Generation Dependencies (Requires `pip install reportlab`) ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except ImportError:
    logging.warning("ReportLab not installed. PDF generation will be disabled. "
                    "Run 'pip install reportlab'")
    SimpleDocTemplate = None

class NSSReportGenerator:
    """
    Generates a consolidated report from the pipeline's findings.
    """

    def __init__(self, config: Dict):
        """
        Initialize the report generator.
        
        Args:
            config: The resolved configuration for the executed pipeline.
        """
        self.config = config
        self.logger = setup_logging(__name__)
        self.survey_type = config.get('survey_type_detected', 'Unknown')
        
        if SimpleDocTemplate:
            self.styles = self._setup_pdf_styles()
        

    def generate_json_report(self, 
                             risk_report: Dict, 
                             privacy_report: Dict,
                             utility_report: Dict
                            ) -> Dict[str, Any]:
        """
        Generates the final comprehensive JSON report intended for the frontend.
        
        Args:
            risk_report: The output from Module 1 (NSSRiskAssessor).
            privacy_report: The summary output from Module 2 (NSSPrivacyEnhancer).
            utility_report: The output from Module 3 (NSSUtilityAssessor).
            
        Returns:
            A consolidated dictionary (JSON report).
        """
        self.logger.info("Generating final consolidated JSON report...")
        
        # Extract key metrics for the summary dashboard
        final_risk = utility_report.get('anonymized_data_risk', {})
        strategy = privacy_report.get('anonymization_strategy', 'unknown')
        
        summary = {
            "pipeline_status": "Success",
            "survey_type": self.survey_type,
            "survey_name": self.config.get('surveys', {}).get(self.survey_type, {}).get('survey_name', 'N/A'),
            "original_records": risk_report.get('risk_metrics', {}).get('total_records', 'N/A'),
            "anonymization_strategy": strategy,
            "final_k_anonymity": final_risk.get('final_k_anonymity', 'N/A'),
            "vulnerable_records_remaining": final_risk.get('vulnerable_records_k1', 'N/A'),
            "final_re_id_risk_percent": final_risk.get('reidentification_rate_percent', 'N/A'),
            "ml_utility_score": utility_report.get('ml_model_comparison', {}).get('anonymized_score (r2_score)', 'N/A'),
            "ml_performance_drop_percent": utility_report.get('ml_model_comparison', {}).get('performance_drop_percent', 'N/A')
        }
        
        # Build the full report
        final_report = {
            "summary": summary,
            "initial_risk_assessment": risk_report,
            "privacy_enhancement_details": privacy_report,
            "utility_assessment": utility_report
        }
        
        return final_report

    def save_pdf_report(self, report_data: Dict, output_path: str):
        """
        Saves the consolidated report as a human-readable PDF.
        
        Args:
            report_data: The JSON report data from generate_json_report.
            output_path: The file path to save the PDF (e.g., /output/report.pdf).
        """
        if not SimpleDocTemplate:
            self.logger.error("Cannot generate PDF. ReportLab library is not installed.")
            return

        self.logger.info(f"Generating PDF report at: {output_path}")
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                    rightMargin=inch, leftMargin=inch,
                                    topMargin=inch, bottomMargin=inch)
            story = []
            
            # --- 1. Title ---
            summary = report_data.get('summary', {})
            story.append(Paragraph(f"NSS SafeData Pipeline: Anonymization Report", self.styles['Title']))
            story.append(Paragraph(f"Survey: {summary.get('survey_name', self.survey_type)}", self.styles['h2']))
            story.append(Spacer(1, 0.25 * inch))

            # --- 2. Executive Summary ---
            story.append(Paragraph("Executive Summary", self.styles['h1']))
            story.append(self._build_summary_table(summary))
            story.append(Spacer(1, 0.25 * inch))

            # --- 3. Initial Risk Assessment (Module 1) ---
            story.append(Paragraph("Initial Risk Assessment (Module 1)", self.styles['h1']))
            risk_report = report_data.get('initial_risk_assessment', {})
            risk_json = json.dumps(risk_report, indent=2, default=str)
            story.append(Preformatted(risk_json, self.styles['Code']))
            story.append(Spacer(1, 0.25 * inch))
            
            # --- 4. Privacy Enhancement (Module 2) ---
            story.append(Paragraph("Privacy Enhancement (Module 2)", self.styles['h1']))
            privacy_report = report_data.get('privacy_enhancement_details', {})
            privacy_json = json.dumps(privacy_report, indent=2, default=str)
            story.append(Preformatted(privacy_json, self.styles['Code']))
            story.append(Spacer(1, 0.25 * inch))

            # --- 5. Utility Assessment (Module 3) ---
            story.append(Paragraph("Utility Assessment (Module 3)", self.styles['h1']))
            utility_report = report_data.get('utility_assessment', {})
            utility_json = json.dumps(utility_report, indent=2, default=str)
            story.append(Preformatted(utility_json, self.styles['Code']))

            # --- Build PDF ---
            doc.build(story)
            self.logger.info("PDF report successfully generated.")

        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}", exc_info=True)

    def _setup_pdf_styles(self) -> Dict:
        """Helper to create ReportLab ParagraphStyles."""
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='Title',
            parent=styles['h1'],
            fontSize=20,
            alignment=1, # Center
            spaceAfter=14
        ))
        
        styles.add(ParagraphStyle(
            name='h1',
            parent=styles['h1'],
            fontSize=16,
            spaceAfter=12,
            borderPadding=4,
            backColor=colors.HexColor("#4682B4"),
            textColor=colors.white
        ))

        styles.add(ParagraphStyle(
            name='h2',
            parent=styles['h2'],
            fontSize=14,
            alignment=1, # Center
            spaceAfter=10
        ))

        styles.add(ParagraphStyle(
            name='Code',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=8,
            borderPadding=6,
            backColor=colors.HexColor("#F0F0F0"),
            textColor=colors.HexColor("#333333"),
            wordWrap='PRE'
        ))
        
        styles.add(ParagraphStyle(
            name='SummaryText',
            parent=styles['Normal'],
            fontSize=10,
            leading=14
        ))
        
        return styles

    def _build_summary_table(self, summary: Dict) -> Paragraph:
        """Helper to format the summary section for the PDF."""
        
        text = f"""
        <b>Pipeline Status:</b> {summary.get('pipeline_status')}
        <br/><b>Anonymization Strategy:</b> {summary.get('anonymization_strategy')}
        <br/>
        <br/><b>Original Records:</b> {summary.get('original_records')}
        <br/><b>Final K-Anonymity:</b> <font color='{"red" if summary.get('final_k_anonymity', 0) < 5 else "green"}'>
            {summary.get('final_k_anonymity')}
        </font>
        <br/><b>Vulnerable Records (k=1) Remaining:</b> <font color='{"red" if summary.get('vulnerable_records_remaining', 0) > 0 else "green"}'>
            {summary.get('vulnerable_records_remaining')}
        </font>
        <br/><b>Final Re-identification Risk:</b> {summary.get('final_re_id_risk_percent', 'N/A')}%
        <br/>
        <br/><b>ML Utility (R² Score):</b> {float(summary.get('ml_utility_score', 0)):.4f}
        <br/><b>Performance Drop:</b> {float(summary.get('ml_performance_drop_percent', 0)):.2f}%
        """
        return Paragraph(text, self.styles['SummaryText'])