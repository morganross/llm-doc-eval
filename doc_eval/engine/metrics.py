import pandas as pd
import numpy as np

class Metrics:
    def __init__(self, outlier_threshold=2.0):
        self.outlier_threshold = outlier_threshold

    def calculate_single_doc_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean, standard deviation, and Z-score based outlier flags
        for single-document evaluation results.

        Args:
            df (pd.DataFrame): DataFrame containing single-document results
                               with columns: doc_id, model, trial, criterion, score.

        Returns:
            pd.DataFrame: DataFrame with aggregated metrics including mean, std_dev,
                          and outlier flags.
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure 'score' is numeric
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['score'], inplace=True)

        # Calculate mean and std-dev per doc_id, criterion, and model
        grouped_metrics = df.groupby(['doc_id', 'criterion', 'model'])['score'].agg(
            mean_score='mean',
            std_dev_score='std'
        ).reset_index()

        # Calculate Z-score for each individual score relative to its group mean and std-dev
        # Merge grouped metrics back to original DataFrame to calculate Z-score
        df_with_metrics = pd.merge(df, grouped_metrics, on=['doc_id', 'criterion', 'model'], how='left')
        
        # Handle cases where std_dev_score is 0 (e.g., only one score or all scores are identical)
        df_with_metrics['z_score'] = df_with_metrics.apply(
            lambda row: (row['score'] - row['mean_score']) / row['std_dev_score'] if row['std_dev_score'] != 0 else 0,
            axis=1
        )
        
        df_with_metrics['is_outlier'] = np.abs(df_with_metrics['z_score']) > self.outlier_threshold

        # Aggregate to show mean, std-dev, and if any trial was an outlier for that doc/criterion/model
        final_metrics = df_with_metrics.groupby(['doc_id', 'criterion', 'model']).agg(
            mean_score=('score', 'mean'),
            std_dev_score=('score', 'std'),
            has_outlier=('is_outlier', 'any')
        ).reset_index()

        return final_metrics

    def calculate_pairwise_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates win rates for pairwise evaluation results.

        Args:
            df (pd.DataFrame): DataFrame containing pairwise results
                               with columns: doc_id_1, doc_id_2, model, winner_doc_id.

        Returns:
            pd.DataFrame: DataFrame with win rates per model and document.
        """
        if df.empty:
            return pd.DataFrame()

        # Calculate total comparisons for each model
        total_comparisons = df.groupby('model').size().reset_index(name='total_comparisons')

        # Calculate wins for each document per model
        # A win means winner_doc_id matches doc_id_1 or doc_id_2
        wins_df = df.groupby(['model', 'winner_doc_id']).size().reset_index(name='wins')

        # Merge to get total comparisons for win rate calculation
        win_rates = pd.merge(wins_df, total_comparisons, on='model', how='left')
        win_rates['win_rate'] = (win_rates['wins'] / win_rates['total_comparisons']) * 100

        # Rename winner_doc_id to doc_id for clarity in output
        win_rates.rename(columns={'winner_doc_id': 'doc_id'}, inplace=True)

        return win_rates[['model', 'doc_id', 'wins', 'total_comparisons', 'win_rate']]

if __name__ == '__main__':
    metrics_calculator = Metrics(outlier_threshold=2.0)

    # --- Test Single Document Metrics ---
    print("--- Testing Single Document Metrics ---")
    single_doc_data = {
        'doc_id': ['doc1', 'doc1', 'doc1', 'doc1', 'doc1', 'doc1', 'doc2', 'doc2', 'doc2', 'doc2', 'doc2', 'doc2'],
        'model': ['ModelA', 'ModelA', 'ModelA', 'ModelB', 'ModelB', 'ModelB', 'ModelA', 'ModelA', 'ModelA', 'ModelB', 'ModelB', 'ModelB'],
        'trial': [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        'criterion': ['Clarity', 'Clarity', 'Coherence', 'Clarity', 'Clarity', 'Coherence', 'Clarity', 'Clarity', 'Coherence', 'Clarity', 'Clarity', 'Coherence'],
        'score': [4, 5, 3, 4, 4, 2, 5, 5, 4, 3, 3, 3],
        'reason': ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12'],
        'timestamp': [pd.Timestamp.now()] * 12
    }
    single_doc_df = pd.DataFrame(single_doc_data)
    single_doc_metrics = metrics_calculator.calculate_single_doc_metrics(single_doc_df)
    print("Single Document Metrics:")
    print(single_doc_metrics)
    print("\n")

    # --- Test Pairwise Win Rates ---
    print("--- Testing Pairwise Win Rates ---")
    pairwise_data = {
        'doc_id_1': ['docA', 'docA', 'docB', 'docA', 'docA', 'docB'],
        'doc_id_2': ['docB', 'docC', 'docC', 'docB', 'docC', 'docC'],
        'model': ['ModelA', 'ModelA', 'ModelA', 'ModelB', 'ModelB', 'ModelB'],
        'trial': [1, 1, 1, 1, 1, 1],
        'winner_doc_id': ['docA', 'docC', 'docB', 'docB', 'docA', 'docC'],
        'reason': ['w1', 'w2', 'w3', 'w4', 'w5', 'w6'],
        'timestamp': [pd.Timestamp.now()] * 6
    }
    pairwise_df = pd.DataFrame(pairwise_data)
    pairwise_win_rates = metrics_calculator.calculate_pairwise_win_rates(pairwise_df)
    print("Pairwise Win Rates:")
    print(pairwise_win_rates)