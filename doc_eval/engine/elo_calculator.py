import pandas as pd
from sqlalchemy import create_engine
from elo.elo import Elo, Rating, WIN, LOSS, DRAW

def calculate_elo_ratings(db_path='doc_eval/results.db', initial_elo=1200):
    """
    Calculates Elo ratings for documents based on pairwise evaluation results.

    Args:
        db_path (str): Path to the SQLite database containing pairwise results.
        initial_elo (int): The initial Elo rating for all documents.

    Returns:
        dict: A dictionary mapping doc_id to its final Elo rating.
    """
    db_engine = create_engine(f'sqlite:///{db_path}')
    try:
        pairwise_df = pd.read_sql_table("pairwise_results", db_engine)
    except ValueError:
        print(f"Error: Table 'pairwise_results' not found in {db_path}.")
        return {}

    if pairwise_df.empty:
        print("No pairwise results found to calculate Elo ratings.")
        return {}

    # Get all unique document IDs
    all_doc_ids = pd.concat([pairwise_df['doc_id_1'], pairwise_df['doc_id_2']]).unique()

    # Initialize Elo ratings for all documents
    elo_ratings = {doc_id: Rating(initial_elo) for doc_id in all_doc_ids}

    # Initialize Elo environment
    elo_env = Elo()

    # Iterate through each pairwise comparison
    for index, row in pairwise_df.iterrows():
        doc_id_1 = row['doc_id_1']
        doc_id_2 = row['doc_id_2']
        winner_doc_id = row['winner_doc_id']

        rating1 = elo_ratings[doc_id_1]
        rating2 = elo_ratings[doc_id_2]

        if winner_doc_id == doc_id_1:
            # doc_id_1 won against doc_id_2
            new_rating1, new_rating2 = elo_env.rate_1vs1(rating1, rating2, drawn=False)
        elif winner_doc_id == doc_id_2:
            # doc_id_2 won against doc_id_1
            new_rating2, new_rating1 = elo_env.rate_1vs1(rating2, rating1, drawn=False)
        else:
            # This case should ideally not happen if winner_doc_id is always one of the two
            # For robustness, we can treat it as a draw or skip
            print(f"Warning: Invalid winner_doc_id '{winner_doc_id}' for pair {doc_id_1} vs {doc_id_2}. Skipping.")
            continue
        
        elo_ratings[doc_id_1] = new_rating1
        elo_ratings[doc_id_2] = new_rating2
    
    # Convert Rating objects back to float for easier consumption
    final_elo_ratings = {doc_id: float(rating) for doc_id, rating in elo_ratings.items()}
    return final_elo_ratings
