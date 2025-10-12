import argparse
import asyncio
import csv
import os
import sqlite3
import datetime
from typing import Optional, List, Tuple

from .api import run_pairwise_evaluation, run_single_evaluation, run_evaluation, get_best_report_by_elo, DB_PATH, DOC_PATHS


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _export_table(db_path: str, table: str, out_csv: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        rows = cur.fetchall()
        headers = [d[0] for d in cur.description]
    finally:
        conn.close()

    _ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def _timestamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _auto_export_pairwise(db_path: str, export_dir: str) -> None:
    os.makedirs(export_dir, exist_ok=True)
    ts = _timestamp()
    pairwise_csv = os.path.join(export_dir, f"pairwise_results_{ts}.csv")
    elo_csv = os.path.join(export_dir, f"elo_summary_{ts}.csv")
    _export_table(db_path, "pairwise_results", pairwise_csv)
    ranking = _compute_elo(db_path)
    with open(elo_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "elo"])
        for doc_id, elo in ranking:
            w.writerow([doc_id, f"{elo:.2f}"])


def _auto_export_single(db_path: str, export_dir: str) -> None:
    os.makedirs(export_dir, exist_ok=True)
    ts = _timestamp()
    single_csv = os.path.join(export_dir, f"single_doc_results_{ts}.csv")
    _export_table(db_path, "single_doc_results", single_csv)


def _compute_elo(db_path: str, k_factor: float = 32.0, initial: float = 1000.0) -> List[Tuple[str, float]]:
    conn = sqlite3.connect(db_path)
    ratings = {}
    try:
        cur = conn.execute(
            "SELECT doc_id_1, doc_id_2, winner_doc_id FROM pairwise_results ORDER BY id ASC"
        )
        for doc1, doc2, winner in cur.fetchall():
            if doc1 not in ratings:
                ratings[doc1] = initial
            if doc2 not in ratings:
                ratings[doc2] = initial
            r1 = ratings[doc1]
            r2 = ratings[doc2]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 - e1
            if winner == doc1:
                s1, s2 = 1.0, 0.0
            elif winner == doc2:
                s1, s2 = 0.0, 1.0
            else:
                s1 = s2 = 0.5
            ratings[doc1] = r1 + k_factor * (s1 - e1)
            ratings[doc2] = r2 + k_factor * (s2 - e2)
    finally:
        conn.close()
    # sort descending by rating
    return sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)


def cmd_run_pairwise(args: argparse.Namespace) -> None:
    docs_dir = args.docs
    db_path = args.db or DB_PATH
    config_path = args.config
    criteria_path = args.criteria

    if not os.path.isdir(docs_dir):
        raise SystemExit(f"--docs folder does not exist: {docs_dir}")

    async def _run():
        await run_pairwise_evaluation(
            folder_path=docs_dir,
            db_path=db_path,
            config_path=config_path,
            criteria_path=criteria_path,
        )

    asyncio.run(_run())
    print(f"Pairwise evaluation complete. DB at: {db_path}")

    # Optionally print winner
    best = get_best_report_by_elo(db_path=db_path, doc_paths=DOC_PATHS)
    if best:
        print(f"Top-ranked candidate (Elo): {best}")
    else:
        print("No winner determined (insufficient data or empty folder).")

    # Auto-export CSVs unless suppressed
    if not getattr(args, "no_export", False):
        export_dir = args.export_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "exports")
        try:
            _auto_export_pairwise(db_path, export_dir)
            print(f"Exported pairwise CSVs to: {export_dir}")
        except Exception as ex:
            print(f"CSV export failed: {ex}")


def cmd_run_single(args: argparse.Namespace) -> None:
    docs_dir = args.docs
    db_path = args.db or DB_PATH
    config_path = args.config
    criteria_path = args.criteria

    if not os.path.isdir(docs_dir):
        raise SystemExit(f"--docs folder does not exist: {docs_dir}")

    async def _run():
        await run_single_evaluation(
            folder_path=docs_dir,
            db_path=db_path,
            config_path=config_path,
            criteria_path=criteria_path,
        )

    asyncio.run(_run())
    print(f"Single-document evaluation complete. DB at: {db_path}")

    # Auto-export CSV unless suppressed
    if not getattr(args, "no_export", False):
        export_dir = args.export_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "exports")
        try:
            _auto_export_single(db_path, export_dir)
            print(f"Exported single-doc CSV to: {export_dir}")
        except Exception as ex:
            print(f"CSV export failed: {ex}")


def cmd_run_both(args: argparse.Namespace) -> None:
    docs_dir = args.docs
    db_path = args.db or DB_PATH
    config_path = args.config
    criteria_path = args.criteria

    if not os.path.isdir(docs_dir):
        raise SystemExit(f"--docs folder does not exist: {docs_dir}")

    async def _run():
        await run_evaluation(
            folder_path=docs_dir,
            mode="both",
            db_path=db_path,
            config_path=config_path,
            criteria_path=criteria_path,
        )

    asyncio.run(_run())
    print(f"Single + Pairwise evaluation complete. DB at: {db_path}")

    # Print winner from pairwise (if available)
    best = get_best_report_by_elo(db_path=db_path, doc_paths=DOC_PATHS)
    if best:
        print(f"Top-ranked candidate (Elo): {best}")
    else:
        print("No winner determined from pairwise (insufficient data or empty folder).")

    # Auto-export CSVs unless suppressed
    if not getattr(args, "no_export", False):
        export_dir = args.export_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "exports")
        try:
            # Try both single and pairwise exports in case both ran
            try:
                _auto_export_single(db_path, export_dir)
            except Exception:
                pass
            try:
                _auto_export_pairwise(db_path, export_dir)
            except Exception:
                pass
            print(f"Exported CSVs to: {export_dir}")
        except Exception as ex:
            print(f"CSV export failed: {ex}")


def cmd_summary(args: argparse.Namespace) -> None:
    db_path = args.db or DB_PATH
    out_csv = args.out

    if not os.path.exists(db_path):
        raise SystemExit(f"--db not found: {db_path}")

    ranking = _compute_elo(db_path)
    if out_csv:
        _ensure_parent_dir(out_csv)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["doc_id", "elo"])
            for doc_id, elo in ranking:
                w.writerow([doc_id, f"{elo:.2f}"])
        print(f"Wrote Elo summary: {out_csv}")
    else:
        print("Elo ranking:")
        for doc_id, elo in ranking:
            print(f"  {doc_id}: {elo:.2f}")


def cmd_export(args: argparse.Namespace) -> None:
    db_path = args.db or DB_PATH
    table = args.table
    out_csv = args.out

    if not os.path.exists(db_path):
        raise SystemExit(f"--db not found: {db_path}")
    if not table:
        raise SystemExit("--table is required (e.g., pairwise_results or single_doc_results)")
    if not out_csv:
        raise SystemExit("--out is required (CSV file path)")

    _export_table(db_path, table, out_csv)
    print(f"Exported {table} to {out_csv}")


def main():
    p = argparse.ArgumentParser(prog="llm_doc_eval.cli", description="LLM Doc Eval CLI (FPF-backed, bootstrap fallback)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run-pairwise
    p_run = sub.add_parser("run-pairwise", help="Run pairwise evaluation over a folder of candidate docs")
    p_run.add_argument("--docs", required=True, help="Path to folder containing candidate report files")
    p_run.add_argument("--db", required=False, help=f"SQLite DB path (default: {DB_PATH})")
    p_run.add_argument("--config", required=False, help="Optional path to config.yaml (for future FPF wiring)")
    p_run.add_argument("--criteria", required=False, help="Optional path to criteria.yaml")
    p_run.add_argument("--export-dir", required=False, help="Directory to write CSV exports (default: <db_dir>/exports)")
    p_run.add_argument("--no-export", action="store_true", help="Do not write CSV exports after run")
    p_run.set_defaults(func=cmd_run_pairwise)

    # run-single
    p_run_single = sub.add_parser("run-single", help="Run single-document grading over a folder of candidate docs")
    p_run_single.add_argument("--docs", required=True, help="Path to folder containing candidate report files")
    p_run_single.add_argument("--db", required=False, help=f"SQLite DB path (default: {DB_PATH})")
    p_run_single.add_argument("--config", required=False, help="Optional path to config.yaml")
    p_run_single.add_argument("--criteria", required=False, help="Optional path to criteria.yaml")
    p_run_single.add_argument("--export-dir", required=False, help="Directory to write CSV exports (default: <db_dir>/exports)")
    p_run_single.add_argument("--no-export", action="store_true", help="Do not write CSV exports after run")
    p_run_single.set_defaults(func=cmd_run_single)

    # run-both
    p_run_both = sub.add_parser("run-both", help="Run single grading and pairwise evaluation in sequence")
    p_run_both.add_argument("--docs", required=True, help="Path to folder containing candidate report files")
    p_run_both.add_argument("--db", required=False, help=f"SQLite DB path (default: {DB_PATH})")
    p_run_both.add_argument("--config", required=False, help="Optional path to config.yaml")
    p_run_both.add_argument("--criteria", required=False, help="Optional path to criteria.yaml")
    p_run_both.add_argument("--export-dir", required=False, help="Directory to write CSV exports (default: <db_dir>/exports)")
    p_run_both.add_argument("--no-export", action="store_true", help="Do not write CSV exports after run")
    p_run_both.set_defaults(func=cmd_run_both)

    # summary
    p_sum = sub.add_parser("summary", help="Compute Elo ranking from pairwise_results")
    p_sum.add_argument("--db", required=False, help=f"SQLite DB path (default: {DB_PATH})")
    p_sum.add_argument("--out", required=False, help="Optional CSV output path for summary")
    p_sum.set_defaults(func=cmd_summary)

    # export
    p_exp = sub.add_parser("export", help="Export a table to CSV")
    p_exp.add_argument("--db", required=False, help=f"SQLite DB path (default: {DB_PATH})")
    p_exp.add_argument("--table", required=True, help="Table name (pairwise_results or single_doc_results)")
    p_exp.add_argument("--out", required=True, help="CSV output path")
    p_exp.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
