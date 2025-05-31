import typer
import pandas as pd
import os
from sqlalchemy import create_engine, text

from doc_eval.loaders.text_loader import load_documents_from_folder
from doc_eval.engine.evaluator import Evaluator
from doc_eval.engine.metrics import Metrics

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
    
    typer.echo("\n--- Raw Single-Document Scores ---")
    typer.echo(df.to_string()) # Display raw scores
    
    typer.echo("\n--- Aggregated Single-Document Summary ---")
    typer.echo(summary_df.to_string()) # Display aggregated summary

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
    Displays a summary of pairwise evaluation results.
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
    summary_df = metrics_calculator.calculate_pairwise_win_rates(df)
    
    typer.echo("\n--- Pairwise Evaluation Summary ---")
    typer.echo(summary_df.to_string())

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

if __name__ == "__main__":
    app()