"""
NSS Utility Assessor (Module 3)
Measures the analytical utility of an anonymized dataset by comparing it
to the original dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import re
from module1_risk_assessment.data_ingestion.utils import setup_logging

class NSSUtilityAssessor:
    """
    Performs a multi-faceted utility analysis by comparing the original
    dataset with its anonymized version.
    """

    def __init__(self, config: Dict):
        """
        Initialize the assessor with the resolved survey-specific configuration.
        """
        self.config = config.get('utility_assessment', {})
        self.logger = setup_logging(__name__)

    def run_utility_analysis(self, 
                             original_df: pd.DataFrame, 
                             anonymized_df: pd.DataFrame,
                             qi_columns: List[str]
                            ) -> Dict[str, Any]:
        """
        Runs the complete utility assessment pipeline.
        
        Args:
            original_df: The original, non-anonymized DataFrame.
            anonymized_df: The anonymized DataFrame (from Module 2).
            qi_columns: The list of quasi-identifiers (from Module 1's report).
            
        Returns:
            A dictionary (the "Utility Report") containing all metrics.
        """
        self.logger.info("==================================================")
        self.logger.info("     === STARTING UTILITY ASSESSMENT (MODULE 3) ===   ")
        self.logger.info("==================================================")
        
        utility_report = {}

        try:
            # 1. Compare basic statistical distributions
            utility_report['statistical_comparison'] = self._compare_basic_statistics(
                original_df, anonymized_df
            )

            # 2. Compare correlation matrices
            utility_report['correlation_comparison'] = self._compare_correlation(
                original_df, anonymized_df
            )

            # 3. Compare Machine Learning model performance
            utility_report['ml_model_comparison'] = self._compare_ml_model_performance(
                original_df, anonymized_df
            )

            # 4. Run a final linkage attack simulation on the *anonymized* data
            utility_report['anonymized_data_risk'] = self._simulate_linkage_attack(
                anonymized_df, qi_columns
            )

            self.logger.info(f"Utility Assessment Report: \n{utility_report}")
            self.logger.info("==================================================")
            self.logger.info("    === UTILITY ASSESSMENT COMPLETED ===    ")
            self.logger.info("==================================================")
            
            return utility_report

        except Exception as e:
            self.logger.error(f"Utility analysis failed: {e}", exc_info=True)
            return {"error": str(e)}

    # --- Task 1: Basic Statistics Comparison ---

    def _preprocess_anonymized_column(self, anon_series: pd.Series) -> pd.Series:
        """
        Converts a potentially generalized anonymized column back to numeric
        for statistical comparison.
        - '20-30' -> 25 (midpoint)
        - '50'    -> 50
        - '*'     -> NaN
        """
        
        def convert_value(val):
            val = str(val)
            # Check for range pattern (e.g., '20-30', '20.5-30.1')
            match = re.match(r'^([\d\.]+)-([\d\.]+)$', val)
            if match:
                try:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    return (lower + upper) / 2
                except ValueError:
                    return np.nan
            # Check for simple numeric pattern
            try:
                return float(val)
            except ValueError:
                return np.nan # For '*', '|', or other non-numeric values

        return anon_series.apply(convert_value).astype(float)

    def _compare_basic_statistics(self, 
                                  original_df: pd.DataFrame, 
                                  anonymized_df: pd.DataFrame
                                 ) -> Dict[str, Any]:
        """
        Compares mean, median, and std dev for key numeric columns.
        """
        self.logger.info("Running statistical comparison...")
        report = {}
        numeric_cols = self.config.get('numeric_columns_to_compare', [])
        
        if not numeric_cols:
            self.logger.warning("No 'numeric_columns_to_compare' specified in config.")
            return {"status": "skipped"}

        for col in numeric_cols:
            if col not in original_df.columns or col not in anonymized_df.columns:
                self.logger.warning(f"Column '{col}' not in both DataFrames. Skipping.")
                continue

            try:
                # Original stats
                orig_stats = original_df[col].describe()
                
                # Preprocess anonymized column and get stats
                anon_col_numeric = self._preprocess_anonymized_column(anonymized_df[col])
                anon_stats = anon_col_numeric.describe()

                # Calculate percentage difference
                def pct_diff(orig, anon):
                    if orig == 0:
                        return np.inf if anon != 0 else 0.0
                    return ((anon - orig) / orig) * 100

                report[col] = {
                    'original_mean': orig_stats.get('mean', np.nan),
                    'anonymized_mean': anon_stats.get('mean', np.nan),
                    'mean_diff_percent': pct_diff(orig_stats.get('mean'), anon_stats.get('mean')),
                    
                    'original_median': orig_stats.get('50%', np.nan),
                    'anonymized_median': anon_stats.get('50%', np.nan),
                    'median_diff_percent': pct_diff(orig_stats.get('50%'), anon_stats.get('50%')),

                    'original_std': orig_stats.get('std', np.nan),
                    'anonymized_std': anon_stats.get('std', np.nan),
                    'std_diff_percent': pct_diff(orig_stats.get('std'), anon_stats.get('std')),
                }
            except Exception as e:
                self.logger.error(f"Failed to compare stats for column '{col}': {e}")
                report[col] = {"error": str(e)}
        
        return report

    # --- Task 2: Correlation Comparison ---

    def _compare_correlation(self, 
                             original_df: pd.DataFrame, 
                             anonymized_df: pd.DataFrame
                            ) -> Dict[str, Any]:
        """
        Calculates the difference between correlation matrices.
        """
        self.logger.info("Running correlation comparison...")
        report = {}
        numeric_cols = self.config.get('numeric_columns_to_compare', [])
        
        if not numeric_cols or len(numeric_cols) < 2:
            self.logger.warning("Correlation comparison requires at least 2 numeric columns. Skipping.")
            return {"status": "skipped"}

        try:
            # Prepare original data
            original_numeric_df = original_df[numeric_cols].copy()
            
            # Prepare anonymized data
            anonymized_numeric_df = pd.DataFrame()
            for col in numeric_cols:
                anonymized_numeric_df[col] = self._preprocess_anonymized_column(anonymized_df[col])

            # Drop NaNs for accurate correlation
            original_numeric_df = original_numeric_df.dropna()
            anonymized_numeric_df = anonymized_numeric_df.dropna()

            if original_numeric_df.empty or anonymized_numeric_df.empty:
                self.logger.error("Not enough valid data to compare correlation.")
                return {"error": "No valid data for correlation"}

            # Calculate matrices
            original_corr = original_numeric_df.corr()
            anonymized_corr = anonymized_numeric_df.corr()

            # Calculate Mean Absolute Difference
            corr_diff = (original_corr - anonymized_corr).abs()
            mean_abs_diff = np.mean(corr_diff.values[np.triu_indices_from(corr_diff, k=1)])
            
            report = {
                'mean_absolute_correlation_difference': mean_abs_diff,
                # Note: Returning full matrices in the report might be too large.
                # 'original_matrix': original_corr.to_dict(),
                # 'anonymized_matrix': anonymized_corr.to_dict(),
            }

        except Exception as e:
            self.logger.error(f"Failed to compare correlation: {e}")
            report = {"error": str(e)}

        return report

    # --- Task 3: Machine Learning Model Performance ---

    ###
    #   Step-by-Step Guide: How We Test ML Utility
    #   (As requested, here is a detailed breakdown of the logic)
    #
    #   **1. The Goal: Why do this?**
    #   This is the most powerful utility test. We want to know if the anonymized
    #   data is still useful for real-world analysis, like building a
    #   predictive model. For example, "Can we still predict a person's
    #   expenditure based on their Age, Sex, and Household Size?"
    #
    #   **2. The "Ground Truth" Score (Original Data)**
    #   First, we build a simple ML model (like Linear Regression) using the
    #   *original* data. We train it to predict a `target_column` (e.g.,
    #   'Monthly_Consumer_Expenditure') using a set of `feature_columns` (e.g.,
    #   ['Age', 'Sex']).
    #   This model gives us a "baseline" or "ground truth" performance score
    #   (e.g., an R-squared score of 0.65). This is the best possible
    #   score we can get with this simple model.
    #
    #   **3. The "Anonymized" Score (New Data)**
    #   Second, we train the *exact same* model type, using the *exact same*
    #   settings, on the *anonymized* data.
    #   This will give us a new performance score (e.g., an R-squared score
    #   of 0.58).
    #
    #   **4. The Comparison: Calculating "Utility Loss"**
    #   Finally, we just compare the two scores.
    #   - Original Score: 0.65
    #   - Anonymized Score: 0.58
    #   - Performance Drop: ((0.65 - 0.58) / 0.65) = 0.107, or 10.7%
    #   We can then report to the user: "The anonymized data retains 89.3%
    #   of the original data's predictive power for this task."
    #
    #   **5. Handling Data Types (The Tricky Part)**
    #   ML models *only* understand numbers. They don't understand '20-30'
    #   or 'Male'.
    #   - **Preprocessing:** We use the `_prepare_data_for_ml` helper function to
    #     clean *both* datasets in the same way.
    #   - **Categorical Features (e.g., 'Sex'):** We use "One-Hot Encoding"
    #     (via `pd.get_dummies`). This turns a column 'Sex' (with values
    #     1 and 2) into two columns: 'Sex_1' (with 1s/0s) and 'Sex_2' (with 1s/0s).
    #   - **Numeric Features (e.g., 'Age'):** We use the `_preprocess_anonymized_column`
    #     function to turn ranges like '20-30' into their midpoints (25).
    #   - **Column Alignment:** We *must* ensure both datasets have the exact
    #     same feature columns after encoding. We use `X_anon.reindex(columns=X_orig.columns)`
    #     to guarantee this, filling any missing columns with 0.
    ###
    
    def _prepare_data_for_ml(self, 
                             df: pd.DataFrame, 
                             feature_cols: List[str], 
                             target_col: str
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares a DataFrame for an ML task.
        - One-hot encodes categorical features.
        - Preprocesses (midpoints) numeric features and target.
        - Drops rows with NaN values.
        """
        
        # Identify which features are numeric vs. categorical from the config
        all_numeric_cols = self.config.get('numeric_columns_to_compare', [])
        
        numeric_features = [col for col in feature_cols if col in all_numeric_cols]
        categorical_features = [col for col in feature_cols if col not in all_numeric_cols]

        # Create the feature DataFrame X
        X = pd.DataFrame(index=df.index)
        
        # Process numeric features
        for col in numeric_features:
            X[col] = self._preprocess_anonymized_column(df[col])
            
        # Process categorical features using One-Hot Encoding
        if categorical_features:
            # We use `astype(str)` to handle any mixed types robustly
            X_cat = pd.get_dummies(df[categorical_features].astype(str), 
                                   columns=categorical_features, 
                                   dummy_na=True)
            X = pd.concat([X, X_cat], axis=1)

        # Process the target column y
        y = self._preprocess_anonymized_column(df[target_col])

        # Drop any rows with missing values in X or y
        Xy = pd.concat([X, y], axis=1).dropna()
        X = Xy[X.columns]
        y = Xy[y.name]
        
        return X, y

    def _compare_ml_model_performance(self, 
                                      original_df: pd.DataFrame, 
                                      anonymized_df: pd.DataFrame
                                     ) -> Dict[str, Any]:
        """
        Trains the same simple ML model on both datasets and compares performance.
        """
        self.logger.info("Running ML model performance comparison...")
        ml_config = self.config.get('ml_model', {})
        
        if not ml_config:
            self.logger.warning("No 'ml_model' config found. Skipping.")
            return {"status": "skipped"}
            
        try:
            task_type = ml_config['task_type']
            target_col = ml_config['target_column']
            feature_cols = ml_config['feature_columns']

            # --- 1. Prepare Original Data ---
            X_orig, y_orig = self._prepare_data_for_ml(original_df, 
                                                       feature_cols, 
                                                       target_col)
            
            # --- 2. Prepare Anonymized Data ---
            X_anon, y_anon = self._prepare_data_for_ml(anonymized_df, 
                                                       feature_cols, 
                                                       target_col)

            if X_orig.empty or X_anon.empty:
                self.logger.error("Not enough valid data for ML comparison.")
                return {"error": "Insufficient valid data after preprocessing"}
            
            # --- 3. Align Columns ---
            # Ensure X_anon has the same columns as X_orig.
            # This handles cases where anonymization (e.g., suppression)
            # might have removed a category.
            X_orig, X_anon = X_orig.align(X_anon, join='left', axis=1, fill_value=0)
            # Align in reverse to ensure X_orig columns are the superset
            X_anon, X_orig = X_anon.align(X_orig, join='left', axis=1, fill_value=0)

            # --- 4. Select Model & Metric ---
            if task_type == 'regression':
                model = LinearRegression()
                score_func = r2_score
                metric_name = 'r2_score'
            elif task_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
                score_func = accuracy_score
                metric_name = 'accuracy_score'
            else:
                raise ValueError(f"Unknown ML task type: {task_type}")

            # --- 5. Train and Score Original ---
            model.fit(X_orig, y_orig)
            preds_orig = model.predict(X_orig)
            score_original = score_func(y_orig, preds_orig)

            # --- 6. Train and Score Anonymized ---
            model.fit(X_anon, y_anon)
            preds_anon = model.predict(X_anon)
            score_anonymized = score_func(y_anon, preds_anon)
            
            # --- 7. Calculate Utility Loss ---
            if abs(score_original) < 1e-6: # Avoid division by zero
                performance_drop_percent = 0.0 if abs(score_anonymized) < 1e-6 else -np.inf
            else:
                performance_drop_percent = ((score_original - score_anonymized) / score_original) * 100

            return {
                'task_type': task_type,
                'model': model.__class__.__name__,
                'target_column': target_col,
                'feature_columns': feature_cols,
                f'original_score ({metric_name})': score_original,
                f'anonymized_score ({metric_name})': score_anonymized,
                'performance_drop_percent': performance_drop_percent
            }

        except Exception as e:
            self.logger.error(f"Failed to compare ML performance: {e}")
            return {"error": str(e)}

    # --- Task 4: Final Linkage Attack Simulation ---
    
    def _simulate_linkage_attack(self, 
                                 anonymized_df: pd.DataFrame, 
                                 qi_columns: List[str]
                                ) -> Dict[str, Any]:
        """
        Simulates a linkage attack on the ANONYMIZED data to test its
        final resilience. This re-uses the logic from Module 1.
        """
        self.logger.info(f"Running linkage attack simulation on *anonymized* data...")

        if not qi_columns:
            self.logger.warning("No QIs provided for attack simulation.")
            return {"status": "skipped", "error": "No QIs provided"}
            
        try:
            # Find the size of the smallest equivalence class (k-anonymity)
            # in the *final* dataset.
            equivalence_classes = anonymized_df.groupby(qi_columns).size()
            final_k = int(equivalence_classes.min())
            
            # Find the number of records that are still unique (k=1)
            vulnerable_records_k1 = int((equivalence_classes == 1).sum())
            
            # Calculate re-identification rate
            total_records = len(anonymized_df)
            if total_records == 0:
                re_id_rate_percent = 0.0
            else:
                # This is the "attacker's risk"
                re_id_rate_percent = (vulnerable_records_k1 / total_records) * 100

            self.logger.info(f"Final k-anonymity score: {final_k}")
            self.logger.info(f"Vulnerable (k=1) records remaining: {vulnerable_records_k1}")
            self.logger.info(f"Re-identification rate: {re_id_rate_percent:.2f}%")

            return {
                'final_k_anonymity': final_k,
                'vulnerable_records_k1': vulnerable_records_k1,
                'reidentification_rate_percent': round(re_id_rate_percent, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to simulate linkage attack: {e}")
            return {"error": str(e)}