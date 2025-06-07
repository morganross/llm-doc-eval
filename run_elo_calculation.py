import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Add the parent directory to the Python path to import elo_calculator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'engine')))

from elo_calculator import calculate_elo_ratings

db_path = 'results.db'
print(f"Attempting to calculate Elo ratings using: {db_path}")
elo_scores = calculate_elo_ratings(db_path=db_path)


print("Final Elo Ratings:")
if elo_scores:
    for doc_id, elo in elo_scores.items():
        print(f"  {doc_id}: {elo:.2f}")
else:
    print("  No Elo ratings to display.")