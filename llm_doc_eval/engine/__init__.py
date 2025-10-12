"""
llm_doc_eval.engine

Engine package for schemas, judge backend, and evaluator orchestration.
Initial implementation provides a deterministic fallback judge to enable
end-to-end testing before FPF wiring is completed.
"""
__all__ = ["schemas", "judge_backend", "evaluator"]
