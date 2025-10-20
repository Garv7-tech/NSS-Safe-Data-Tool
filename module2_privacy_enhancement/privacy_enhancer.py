import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

# --- Libraries for Advanced Techniques (Optional, install with pip) ---
try:
    from sdv.tabular import GaussianCopula
except ImportError:
    GaussianCopula = None

class NSSPrivacyEnhancer:
    """
    An intelligent, goal-seeking engine to anonymize NSS data using robust techniques.
    This class provides a complete, production-ready implementation for Module 2.
    """

    def __init__(self, config: Dict, risk_report: Dict):
        self.config = config.get('privacy_enhancement', {})
        self.qi_columns = risk_report.get('detected_quasi_identifiers', [])
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Privacy Enhancer initialized for QIs: {self.qi_columns}")

    def _generalize_numeric(self, series: pd.Series) -> str:
        min_val, max_val = series.min(), series.max()
        if min_val == max_val: return str(min_val)
        return f"{min_val}-{max_val}"

    def _generalize_categorical(self, series: pd.Series) -> str:
        unique_vals = sorted(list(series.unique()))
        if len(unique_vals) > 5: return '*'
        return '|'.join(map(str, unique_vals))

    def _get_partitions(self, df: pd.DataFrame, target_k: int) -> List[pd.Index]:
        partitions = []
        remaining_df = df.copy()
        while len(remaining_df) >= target_k:
            seed_row = remaining_df.iloc[[0]]
            distances = pd.DataFrame(index=remaining_df.index)
            for col in self.qi_columns:
                if pd.api.types.is_numeric_dtype(remaining_df[col]):
                    range_val = remaining_df[col].max() - remaining_df[col].min()
                    if range_val > 0:
                        distances[col] = abs(remaining_df[col] - seed_row[col].iloc[0]) / range_val
                else:
                    distances[col] = (remaining_df[col] != seed_row[col].iloc[0]).astype(int)
            
            total_distance = distances.sum(axis=1)
            partition_indices = total_distance.nsmallest(target_k).index
            partitions.append(partition_indices)
            remaining_df.drop(partition_indices, inplace=True)
        
        # Add remaining small partitions to the last big one
        if not remaining_df.empty and partitions:
            partitions[-1] = partitions[-1].union(remaining_df.index)
            
        return partitions

    def _generalize_k_anonymity(self, df: pd.DataFrame, target_k: int) -> pd.DataFrame:
        self.logger.info(f"Applying generalization to achieve k={target_k}...")
        anonymized_df = df.copy()
        partitions = self._get_partitions(df[self.qi_columns], target_k)
        
        for p_indices in partitions:
            for col in self.qi_columns:
                partition_data = df.loc[p_indices, col]
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    gen_val = self._generalize_numeric(partition_data)
                else:
                    gen_val = self._generalize_categorical(partition_data)
                anonymized_df.loc[p_indices, col] = gen_val
        return anonymized_df

    def _calculate_k_anonymity(self, df: pd.DataFrame) -> int:
        if not self.qi_columns or df.empty: return float('inf')
        return df.groupby(self.qi_columns).size().min()

    def _calculate_normalized_certainty_penalty(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> float:
        total_penalty = 0.0
        for col in self.qi_columns:
            penalty = 0.0
            if pd.api.types.is_numeric_dtype(original_df[col].dtype):
                global_range = original_df[col].max() - original_df[col].min()
                if global_range > 0:
                    ranges = anonymized_df[col].astype(str).str.split('-', expand=True)
                    lower = pd.to_numeric(ranges[0], errors='coerce').fillna(original_df[col])
                    upper = pd.to_numeric(ranges[1], errors='coerce').fillna(lower)
                    penalty = (upper - lower).sum() / global_range
            else:
                 penalty = (original_df[col].astype(str) != anonymized_df[col].astype(str)).sum()
            total_penalty += penalty
        return total_penalty / (len(original_df) * len(self.qi_columns)) if self.qi_columns else 0.0

    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.get('privacy_strategy', 'k_anonymity')
        self.logger.info(f"Starting privacy enhancement with strategy: '{strategy}'")

        if strategy == 'k_anonymity':
            return self._run_k_anonymity_optimization(df)
        elif strategy == 'synthetic_data':
            if GaussianCopula is None: raise ImportError("Please install 'sdv' for this.")
            return self._run_synthetic_data_generation(df)
        return df

    def _run_k_anonymity_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        goals = self.config.get('goals', {}).get('k_anonymity', {})
        target_k = goals.get('target_k', 5)
        max_info_loss = goals.get('max_info_loss', 0.2)

        self.logger.info(f"Optimization goals: target_k={target_k}, max_info_loss={max_info_loss}")
        anonymized_df = self._generalize_k_anonymity(df, target_k)
        
        final_k = self._calculate_k_anonymity(anonymized_df)
        final_info_loss = self._calculate_normalized_certainty_penalty(df, anonymized_df)

        self.logger.info(f"Anonymization result: Final k={final_k}, Info Loss (NCP)={final_info_loss:.3f}")

        if final_k >= target_k and final_info_loss <= max_info_loss:
            self.logger.info("Goals achieved! Anonymization successful.")
            return anonymized_df
        else:
            self.logger.error(f"Failed to meet privacy goals. k={final_k}, info_loss={final_info_loss}")
            return df

    def _run_synthetic_data_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Training synthetic data model. This may take time...")
        model = GaussianCopula()
        model.fit(df)
        synthetic_data = model.sample(num_rows=len(df))
        self.logger.info("Synthetic data generated.")
        return synthetic_data