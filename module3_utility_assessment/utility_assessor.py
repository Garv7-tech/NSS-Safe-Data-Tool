import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
# This import is new and requires 'scipy' to be installed
from scipy.stats import ks_2samp # Kolmogorov-Smirnov test

logger = logging.getLogger(__name__)

class UtilityAssessor:
    """
    Compares the utility of the original vs. anonymized dataset.
    
    New in this version:
    - Includes `_compare_distributions` using KS test to fulfill PS-1.pdf. 
    """
    def __init__(self, 
                 original_df: pd.DataFrame, 
                 anonymized_df: pd.DataFrame, 
                 survey_config: Dict[str, Any]):
        
        self.original_df = original_df.copy()
        self.anonymized_df = anonymized_df.copy()
        self.config = survey_config
        self.report: Dict[str, Any] = {
            'aggregate_statistics': {},
            'distribution_comparison': {}, # New section
            'ml_model_utility': {}
        }

    def _get_common_numeric_cols(self) -> List[str]:
        """Helper to find numeric columns present in both dataframes."""
        try:
            original_numeric = self.original_df.select_dtypes(include=['number']).columns
            anonymized_numeric = self.anonymized_df.select_dtypes(include=['number']).columns
            # We can only compare columns that are still numeric in the anonymized set
            common_cols = list(set(original_numeric) & set(anonymized_numeric))
            
            # Try to find the main expenditure/income column
            sensitive = self.config.get('sensitive_attributes', [])
            for col in sensitive:
                if col in common_cols:
                    # Return a list with the sensitive column first
                    other_cols = [c for c in common_cols if c != col and c not in self.config.get('id_attributes', {}).values()]
                    return [col] + other_cols[:1] # Return sensitive col + 1 other max
            
            # Fallback if no sensitive col found
            return [col for col in common_cols if col not in self.config.get('id_attributes', {}).values()][:2] # Return max 2
        except Exception as e:
            logger.warning(f"Error getting common numeric cols: {e}")
            return []

    def _compare_aggregate_stats(self) -> Dict[str, Any]:
        """Compares mean and std dev of numeric columns."""
        logger.info("Comparing aggregate statistics...")
        stats_report = {}
        numeric_cols = self._get_common_numeric_cols()
        
        if not numeric_cols:
            logger.warning("No common numeric columns found to compare stats.")
            return {'status': 'skipped', 'reason': 'No common numeric columns'}

        for col in numeric_cols:
            try:
                original_mean = self.original_df[col].mean()
                anonymized_mean = self.anonymized_df[col].mean()
                original_std = self.original_df[col].std()
                anonymized_std = self.anonymized_df[col].std()
                
                mean_diff = (anonymized_mean - original_mean) / original_mean if original_mean != 0 else 0
                std_diff = (anonymized_std - original_std) / original_std if original_std != 0 else 0

                stats_report[col] = {
                    'original_mean': round(original_mean, 2),
                    'anonymized_mean': round(anonymized_mean, 2),
                    'mean_percent_diff': round(mean_diff * 100, 2),
                    'original_std': round(original_std, 2),
                    'anonymized_std': round(anonymized_std, 2),
                    'std_percent_diff': round(std_diff * 100, 2)
                }
            except Exception as e:
                logger.warning(f"Could not compare stats for column {col}: {e}")
                stats_report[col] = {'status': 'failed', 'error': str(e)}
        
        return stats_report

    def _compare_distributions(self) -> Dict[str, Any]:
        """
        Compares the distributions of key numeric columns using the
        Kolmogorov-Smirnov (KS) two-sample test.
        """
        logger.info("Comparing column distributions (KS test)...")
        dist_report = {}
        numeric_cols = self._get_common_numeric_cols()

        if not numeric_cols:
            logger.warning("No common numeric columns found to compare distributions.")
            return {'status': 'skipped', 'reason': 'No common numeric columns'}
            
        for col in numeric_cols:
            try:
                # Ensure columns are numeric and drop NAs for the test
                original_series = pd.to_numeric(self.original_df[col], errors='coerce').dropna()
                anonymized_series = pd.to_numeric(self.anonymized_df[col], errors='coerce').dropna()

                if original_series.empty or anonymized_series.empty:
                    dist_report[col] = {'status': 'skipped', 'reason': 'No numeric data'}
                    continue

                # KS Test:
                # p-value > 0.05 suggests the two distributions are similar.
                # p-value < 0.05 suggests they are different.
                statistic, p_value = ks_2samp(original_series, anonymized_series)
                
                dist_report[col] = {
                    'ks_statistic': round(statistic, 4),
                    'p_value': round(p_value, 4),
                    'are_distributions_similar': bool(p_value > 0.05)
                }
            except Exception as e:
                logger.warning(f"Could not compare distribution for column {col}: {e}")
                dist_report[col] = {'status': 'failed', 'error': str(e)}
        
        return dist_report

    def _compare_ml_utility(self) -> Dict[str, Any]:
        """
        Trains a simple ML model on both datasets to see if
        analytical utility is preserved.
        """
        logger.info("Comparing ML model utility...")
        
        # Try to find a good target variable (e.g., expenditure)
        target_col = None
        sensitive_cols = self.config.get('sensitive_attributes', [])
        for col in sensitive_cols:
            if col in self.original_df.columns and pd.api.types.is_numeric_dtype(self.original_df[col]):
                target_col = col
                break
        
        if not target_col:
            logger.warning("No suitable numeric target column found for ML model. Skipping.")
            return {'status': 'skipped', 'reason': 'No numeric sensitive/target column found'}

        # Use QIs as features
        features = [qi for qi in self.config.get('quasi_identifiers', []) if qi in self.original_df.columns]
        if not features:
            logger.warning("No suitable features (QIs) found for ML model. Skipping.")
            return {'status': 'skipped', 'reason': 'No QIs found to use as features'}
            
        logger.info(f"Using ML model to predict '{target_col}' from features {features}")

        # Preprocessing: One-hot encode categorical features
        # Identify categorical features based on *original* data
        categorical_features = [f for f in features if pd.api.types.is_string_dtype(self.original_df[f]) or pd.api.types.is_object_dtype(self.original_df[f])]
        numeric_features = [f for f in features if f not in categorical_features]
        
        # Ensure target is not in features
        if target_col in categorical_features:
            categorical_features.remove(target_col)
        if target_col in numeric_features:
            numeric_features.remove(target_col)
            
        # Update features list to only include valid ones
        features = categorical_features + numeric_features

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='drop') # Drop columns not in features

        # Create the model pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])
                                
        def train_and_evaluate(df: pd.DataFrame, model_name: str) -> Optional[float]:
            try:
                # Prep data
                df_clean = df[features + [target_col]].dropna()
                if len(df_clean) < 100:
                    logger.warning(f"Not enough data for {model_name} model ({len(df_clean)} rows).")
                    return None
                    
                X = df_clean[features]
                y = df_clean[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                logger.info(f"R-squared score for {model_name} model: {r2:.3f}")
                return r2
            except Exception as e:
                # This can happen if generalization created '*' for all values
                logger.error(f"Failed to train/evaluate {model_name} model: {e}", exc_info=True)
                return None

        # Train on original data
        r2_original = train_and_evaluate(self.original_df, "Original")
        
        # Train on anonymized data
        r2_anonymized = train_and_evaluate(self.anonymized_df, "Anonymized")

        if r2_original is not None and r2_anonymized is not None:
            # Calculate information loss
            info_loss = (r2_original - r2_anonymized) / r2_original if r2_original > 0 else 0
            return {
                'status': 'success',
                'target_variable': target_col,
                'features_used': features,
                'r2_original': round(r2_original, 3),
                'r2_anonymized': round(r2_anonymized, 3),
                'performance_drop_percentage': round(info_loss * 100, 2)
            }
        else:
            return {'status': 'failed', 'reason': 'Model training failed on one or both datasets.'}

    def assess_utility(self) -> Dict[str, Any]:
        """Runs all utility assessment tasks."""
        logger.info("--- STAGE 4: UTILITY ASSESSMENT (MODULE 3) ---")
        
        # 1. Compare aggregate stats (existing)
        self.report['aggregate_statistics'] = self._compare_aggregate_stats()
        
        # 2. Compare distributions (new)
        self.report['distribution_comparison'] = self._compare_distributions()
        
        # 3. Compare ML utility (existing)
        self.report['ml_model_utility'] = self._compare_ml_utility()

        logger.info("Utility assessment complete.")
        return self.report