"""
llm_doc_eval package

FPF-backed evaluator skeleton with a stable API surface for ACM:
- DOC_PATHS, DB_PATH constants
- run_pairwise_evaluation(...)
- get_best_report_by_elo(...)

Subpackages:
- engine: evaluator orchestration, schemas, judge backend (FPF or fallback)
- fpf_loader: resolves FilePromptForge path

This initial implementation ships a working evaluator with a deterministic
fallback judge (content-length based) to enable integration testing before
FPF wiring is completed. The SQLite schema and API surface are preserved.
"""
__all__ = [
    "api",
]
