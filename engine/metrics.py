import pandas as pd
import numpy as np

class Metrics:
    def __init__(self, outlier_threshold=2.0):
        self.outlier_threshold = outlier_threshold

    def calculate_single_doc_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean, standard deviation, and Z-score based outlier flags
        for single-document evaluation results.
        Also includes individual trial scores in the aggregated output.

        Args:
            df (pd.DataFrame): DataFrame containing single-document results
                                with columns: doc_id, model, trial, criterion, score.

        Returns:
            pd.DataFrame: DataFrame with aggregated metrics including mean, std_dev,
                          individual trial scores, and outlier flags.
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure 'score' is numeric
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['score'], inplace=True)

        # Calculate mean and std-dev for each group (doc_id, criterion, model)
        grouped_stats = df.groupby(['doc_id', 'criterion', 'model'])['score'].agg(
            mean_score='mean',
            std_dev_score='std'
        ).reset_index()

        # Calculate Z-score and outlier flags for each individual score
        df_with_group_stats = pd.merge(df, grouped_stats, on=['doc_id', 'criterion', 'model'], how='left')
        
        df_with_group_stats['z_score'] = df_with_group_stats.apply(
            lambda row: (row['score'] - row['mean_score']) / row['std_dev_score'] if row['std_dev_score'] != 0 else 0,
            axis=1
        )
        df_with_group_stats['is_outlier'] = np.abs(df_with_group_stats['z_score']) > self.outlier_threshold

        # Determine if any trial for a given doc/criterion/model group was an outlier
        outlier_flags = df_with_group_stats.groupby(['doc_id', 'criterion', 'model'])['is_outlier'].any().reset_index(name='has_outlier')

        # Pivot the original DataFrame to get scores from each trial as separate columns
        # This assumes 'trial' column exists and contains distinct trial numbers
        pivoted_scores = df.pivot_table(
            index=['doc_id', 'criterion', 'model'],
            columns='trial',
            values='score'
        ).add_prefix('score_trial_').reset_index()

        # Merge pivoted scores with mean, std_dev, and outlier flags
        final_metrics = pd.merge(pivoted_scores, grouped_stats, on=['doc_id', 'criterion', 'model'], how='left')
        final_metrics = pd.merge(final_metrics, outlier_flags, on=['doc_id', 'criterion', 'model'], how='left')

        # Reorder columns for clarity
        # Get all unique trial numbers to ensure all score_trial_X columns are included
        all_trial_cols = sorted([col for col in pivoted_scores.columns if col.startswith('score_trial_')])
        
        desired_cols = ['doc_id', 'model', 'criterion'] + all_trial_cols + ['mean_score', 'std_dev_score', 'has_outlier']
        
        # Filter to only include columns that actually exist in final_metrics
        existing_desired_cols = [col for col in desired_cols if col in final_metrics.columns]

        return final_metrics[existing_desired_cols]

    def calculate_overall_average_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the overall average score for each document, averaged over all criteria and trials.

        Args:
            df (pd.DataFrame): DataFrame containing single-document results
                                with columns: doc_id, model, trial, criterion, score.

        Returns:
            pd.DataFrame: DataFrame with doc_id and its overall average score.
        """
        if df.empty:
            return pd.DataFrame()

        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['score'], inplace=True)

        overall_avg_scores = df.groupby('doc_id')['score'].mean().reset_index()
        overall_avg_scores.rename(columns={'score': 'overall_average_score'}, inplace=True)
        return overall_avg_scores


    def calculate_pairwise_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates win rates for pairwise evaluation results, including overall win rates per document.

        Args:
            df (pd.DataFrame): DataFrame containing pairwise results
                                with columns: doc_id_1, doc_id_2, model, winner_doc_id.

        Returns:
            pd.DataFrame: DataFrame with overall win rates per document.
        """
        if df.empty:
            return pd.DataFrame()

        # Calculate overall wins for each document (how many times it was chosen as winner)
        overall_wins = df.groupby('winner_doc_id').size().reset_index(name='overall_wins')
        overall_wins.rename(columns={'winner_doc_id': 'doc_id'}, inplace=True)

        # Calculate overall participations for each document (how many times it appeared in a pair)
        # Create a temporary DataFrame to count participations for each doc_id
        participations_df = pd.DataFrame({
            'doc_id': pd.concat([df['doc_id_1'], df['doc_id_2']])
        })
        overall_participations = participations_df.value_counts().reset_index(name='overall_total_participations')
        overall_participations.rename(columns={'index': 'doc_id'}, inplace=True)

        # Merge wins and participations to calculate overall win rate
        overall_win_rates = pd.merge(overall_wins, overall_participations, on='doc_id', how='right')
        overall_win_rates['overall_wins'] = overall_win_rates['overall_wins'].fillna(0).astype(int)
        overall_win_rates['overall_win_rate'] = (overall_win_rates['overall_wins'] / overall_win_rates['overall_total_participations']) * 100
        
        return overall_win_rates[['doc_id', 'overall_wins', 'overall_total_participations', 'overall_win_rate']]

if __name__ == '__main__':
    metrics_calculator = Metrics(outlier_threshold=2.0)

    # --- Test Single Document Metrics ---
    print("--- Test Single Document Metrics ---")
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

    # --- Test Overall Average Scores ---
    print("--- Test Overall Average Scores ---")
    overall_avg_scores = metrics_calculator.calculate_overall_average_scores(single_doc_df)
    print("Overall Average Scores:")
    print(overall_avg_scores)
    print("\n")

    # --- Test Pairwise Win Rates ---
    print("--- Test Pairwise Win Rates ---")
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