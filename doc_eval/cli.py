import typer
import pandas as pd
import os
from sqlalchemy import create_engine, text

from doc_eval.loaders.text_loader import load_documents_from_folder
from doc_eval.engine.evaluator import Evaluator
from doc_eval.engine.metrics import Metrics
from doc_eval.engine.elo_calculator import calculate_elo_ratings # Import the new function

app = typer.Typer()
DB_PATH = 'doc_eval/results.db'

def _clear_table(table_name: str):
    """Clears all data from the specified table."""
    if not os.path.exists(DB_PATH):
        return # No DB to clear
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    with db_engine.connect() as connection:
        connection.execute(text(f"DELETE FROM {table_name}"))
        connection.commit()
    typer.echo(f"Cleared table: {table_name}")

@app.command()
def run_single(
    folder_path: str = typer.Argument(..., help="Path to the folder containing documents for single-document evaluation.")
):
    """
    Runs single-document evaluations on files in the specified folder.
    """
    typer.echo(f"Running single-document evaluations on: {folder_path}")
    _clear_table("single_doc_results") # Clear table before new run
    evaluator = Evaluator(db_path=DB_PATH)
    
    documents = list(load_documents_from_folder(folder_path))
    if not documents:
        typer.echo("No .txt or .md documents found in the specified folder.")
        return

    for doc_id, content in documents:
        typer.echo(f"  Evaluating single document: {doc_id}")
        evaluator.evaluate_single_document(doc_id, content)
    
    typer.echo("Single-document evaluations complete.")

@app.command()
def summary_single():
    """
    Displays a summary of single-document evaluation results.
    """
    typer.echo("Displaying single-document evaluation summary.")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-single' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("single_doc_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'single_doc_results' not found in {DB_PATH}. Please run 'run-single' first.")
        return
    
    metrics_calculator = Metrics()
    summary_df = metrics_calculator.calculate_single_doc_metrics(df)
    overall_avg_scores_df = metrics_calculator.calculate_overall_average_scores(df) # New line

    typer.echo("\n--- Raw Single-Document Scores ---")
    typer.echo(df.to_string()) # Display raw scores
    
    typer.echo("\n--- Aggregated Single-Document Summary ---")
    typer.echo(summary_df.to_string()) # Display aggregated summary

    typer.echo("\n--- Overall Average Scores Per Document ---") # New line
    typer.echo(overall_avg_scores_df.to_string()) # New line

@app.command()
def run_pairwise(
    folder_path: str = typer.Argument(..., help="Path to the folder containing documents for pairwise evaluation.")
):
    """
    Runs pairwise evaluations on files in the specified folder.
    """
    typer.echo(f"Running pairwise evaluations on: {folder_path}")
    _clear_table("pairwise_results") # Clear table before new run
    evaluator = Evaluator(db_path=DB_PATH)

    documents = list(load_documents_from_folder(folder_path))
    if not documents:
        typer.echo("No .txt or .md documents found in the specified folder.")
        return
    
    typer.echo(f"  Evaluating {len(documents)} documents in pairwise combinations.")
    evaluator.evaluate_pairwise_documents(documents)

    typer.echo("Pairwise evaluations complete.")

@app.command()
def summary_pairwise():
    """
    Displays a summary of pairwise evaluation results, including overall win rates and Elo ratings.
    Also identifies the document with the highest win rate and highest Elo rating.
    """
    typer.echo("Displaying pairwise evaluation summary.")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("pairwise_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'pairwise_results' not found in {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    metrics_calculator = Metrics()
    overall_win_rates_df = metrics_calculator.calculate_pairwise_win_rates(df)
    
    # Calculate Elo ratings
    elo_scores = calculate_elo_ratings(db_path=DB_PATH)
    elo_df = pd.DataFrame.from_dict(elo_scores, orient='index', columns=['elo_rating']).reset_index()
    elo_df.rename(columns={'index': 'doc_id'}, inplace=True)

    # Merge win rates and Elo ratings
    combined_summary_df = pd.merge(overall_win_rates_df, elo_df, on='doc_id', how='left')
    combined_summary_df['elo_rating'] = combined_summary_df['elo_rating'].fillna(0).round(2) # Fill NaN Elo with 0 and round

    typer.echo("\n--- Overall Pairwise Summary (Win Rates & Elo Ratings) ---")
    typer.echo(combined_summary_df.to_string())

    if not combined_summary_df.empty:
        # Document with highest Win Rate
        highest_win_doc = combined_summary_df.loc[combined_summary_df['overall_win_rate'].idxmax()]
        typer.echo(f"\n--- Document with Highest Win Rate ---")
        typer.echo(f"Document ID: {highest_win_doc['doc_id']}")
        typer.echo(f"Overall Win Rate: {highest_win_doc['overall_win_rate']:.2f}%")
        typer.echo(f"Elo Rating: {highest_win_doc['elo_rating']:.2f}")

        # Document with highest Elo Rating
        highest_elo_doc = combined_summary_df.loc[combined_summary_df['elo_rating'].idxmax()]
        typer.echo(f"\n--- Document with Highest Elo Rating ---")
        typer.echo(f"Document ID: {highest_elo_doc['doc_id']}")
        typer.echo(f"Overall Win Rate: {highest_elo_doc['overall_win_rate']:.2f}%")
        typer.echo(f"Elo Rating: {highest_elo_doc['elo_rating']:.2f}")

@app.command()
def raw_pairwise():
    """
    Displays all raw pairwise evaluation results.
    """
    typer.echo("Displaying raw pairwise evaluation results.")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("pairwise_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'pairwise_results' not found in {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    typer.echo("\n--- Raw Pairwise Evaluation Results ---")
    # Drop the 'reason' and 'timestamp' columns as requested
    df_display = df.drop(columns=['reason', 'timestamp'], errors='ignore')
    typer.echo(df_display.to_string())

@app.command()
def export_single_raw(
    output_path: str = typer.Option("single_doc_raw_results.csv", help="Path to save the raw single-document results CSV.")
):
    """
    Exports raw single-document evaluation results to a CSV file.
    """
    typer.echo(f"Exporting raw single-document results to {output_path}")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-single' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("single_doc_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'single_doc_results' not found in {DB_PATH}. Please run 'run-single' first.")
        return
    
    df.to_csv(output_path, index=False)
    typer.echo(f"Raw single-document results exported to {output_path}")

@app.command()
def export_single_summary(
    output_path: str = typer.Option("single_doc_summary.csv", help="Path to save the aggregated single-document summary CSV.")
):
    """
    Exports aggregated single-document evaluation summary to a CSV file.
    """
    typer.echo(f"Exporting aggregated single-document summary to {output_path}")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-single' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("single_doc_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'single_doc_results' not found in {DB_PATH}. Please run 'run-single' first.")
        return
    
    metrics_calculator = Metrics()
    summary_df = metrics_calculator.calculate_single_doc_metrics(df)
    overall_avg_scores_df = metrics_calculator.calculate_overall_average_scores(df) # New line

    # Merge overall average scores into the summary_df for export
    merged_summary_df = pd.merge(summary_df, overall_avg_scores_df, on='doc_id', how='left') # New line
    
    merged_summary_df.to_csv(output_path, index=False) # Changed to merged_summary_df
    typer.echo(f"Aggregated single-document summary exported to {output_path}")

@app.command()
def export_pairwise_raw(
    output_path: str = typer.Option("pairwise_raw_results.csv", help="Path to save the raw pairwise evaluation results CSV.")
):
    """
    Exports raw pairwise evaluation results to a CSV file.
    """
    typer.echo(f"Exporting raw pairwise results to {output_path}")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("pairwise_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'pairwise_results' not found in {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    # Drop the 'reason' and 'timestamp' columns as they are not typically needed in raw exports
    df_display = df.drop(columns=['reason', 'timestamp'], errors='ignore')
    df_display.to_csv(output_path, index=False)
    typer.echo(f"Raw pairwise results exported to {output_path}")

@app.command()
def export_pairwise_summary(
    output_path: str = typer.Option("pairwise_summary.csv", help="Path to save the aggregated pairwise summary CSV.")
):
    """
    Exports aggregated pairwise evaluation summary to a CSV file.
    """
    typer.echo(f"Exporting aggregated pairwise summary to {output_path}")
    if not os.path.exists(DB_PATH):
        typer.echo(f"Error: Database file not found at {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    db_engine = create_engine(f'sqlite:///{DB_PATH}')
    try:
        df = pd.read_sql_table("pairwise_results", db_engine)
    except ValueError:
        typer.echo(f"Error: Table 'pairwise_results' not found in {DB_PATH}. Please run 'run-pairwise' first.")
        return
    
    metrics_calculator = Metrics()
    summary_df = metrics_calculator.calculate_pairwise_win_rates(df)
    
    summary_df.to_csv(output_path, index=False)
    typer.echo(f"Aggregated pairwise summary exported to {output_path}")

if __name__ == "__main__":
    app()