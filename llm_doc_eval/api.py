import os
import sqlite3
import itertools
import datetime
from typing import Dict, Optional, List, Tuple, Any

# Public globals expected by api_cost_multiplier/evaluate.py
DOC_PATHS: Dict[str, str] = {}

# Default DB lives alongside this package
DB_PATH: str = os.path.join(os.path.dirname(__file__), "results.sqlite")

# Engine and wiring (no fallbacks)
from .engine.schemas import load_criteria
from .engine.judge_backend import FPFJudge, JudgeCfg
from .fpf_loader import ensure_fpf
from api_cost_multiplier.functions import fpf_runner  # type: ignore
import tempfile
import json
import uuid as _uuid

try:
    import yaml  # type: ignore
except Exception as _e:
    yaml = None


# --- Internal helpers ---------------------------------------------------------

def _ensure_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Keep schema names identical to legacy for compatibility
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS single_doc_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                model TEXT NOT NULL,
                trial INTEGER NOT NULL,
                criterion TEXT NOT NULL,
                score INTEGER NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pairwise_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id_1 TEXT NOT NULL,
                doc_id_2 TEXT NOT NULL,
                model TEXT NOT NULL,
                trial INTEGER NOT NULL,
                winner_doc_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _read_candidates(folder_path: str, exts: Tuple[str, ...] = (".md", ".txt")) -> Dict[str, str]:
    """
    Scan folder_path for candidate files by extension.
    Returns a mapping: doc_id (filename) -> absolute path.
    """
    paths: Dict[str, str] = {}
    for name in os.listdir(folder_path):
        p = os.path.join(folder_path, name)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            doc_id = name
            # On collision, append a counter suffix to doc_id
            if doc_id in paths:
                base, ext = os.path.splitext(doc_id)
                suffix = 1
                while f"{base}_{suffix}{ext}" in paths:
                    suffix += 1
                doc_id = f"{base}_{suffix}{ext}"
            paths[doc_id] = os.path.abspath(p)
    return paths


def _load_contents(doc_paths: Dict[str, str]) -> Dict[str, str]:
    contents: Dict[str, str] = {}
    for doc_id, p in doc_paths.items():
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                contents[doc_id] = fh.read()
        except Exception:
            contents[doc_id] = ""
    return contents


def _persist_pair_result(
    conn: sqlite3.Connection,
    doc_id_1: str,
    doc_id_2: str,
    winner_doc_id: str,
    model_label: str,
    trial: int,
    reason: str,
) -> None:
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO pairwise_results
            (doc_id_1, doc_id_2, model, trial, winner_doc_id, reason, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id_1, doc_id_2, model_label, trial, winner_doc_id, reason, ts),
    )


def _persist_single_result(
    conn: sqlite3.Connection,
    doc_id: str,
    model_label: str,
    trial: int,
    criterion: str,
    score: int,
    reason: str,
) -> None:
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO single_doc_results
            (doc_id, model, trial, criterion, score, reason, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, model_label, trial, criterion, int(score), reason, ts),
    )


def _compute_elo_from_db(db_path: str, k_factor: float = 32.0, initial: float = 1000.0) -> Dict[str, float]:
    """
    Simple Elo calculator over pairwise_results. Order-independent single pass.
    """
    conn = sqlite3.connect(db_path)
    ratings: Dict[str, float] = {}
    try:
        cur = conn.execute(
            "SELECT doc_id_1, doc_id_2, winner_doc_id FROM pairwise_results ORDER BY id ASC"
        )
        rows = cur.fetchall()
        for doc1, doc2, winner in rows:
            if doc1 not in ratings:
                ratings[doc1] = initial
            if doc2 not in ratings:
                ratings[doc2] = initial
            r1 = ratings[doc1]
            r2 = ratings[doc2]
            # Expected scores
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 - e1
            # Actual scores
            if winner == doc1:
                s1, s2 = 1.0, 0.0
            elif winner == doc2:
                s1, s2 = 0.0, 1.0
            else:
                # Should not happen; treat as tie
                s1 = s2 = 0.5
            # Update
            ratings[doc1] = r1 + k_factor * (s1 - e1)
            ratings[doc2] = r2 + k_factor * (s2 - e2)
    finally:
        conn.close()
    return ratings


async def _jsonify_response(raw_text: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Two-stage JSON recovery: if raw text is not valid JSON, use an LLM (jsonify provider)
    to reformat it into strict JSON.
    
    Args:
        raw_text: The raw response text that failed JSON parsing
        config: The jsonify config section from config.yaml
        
    Returns:
        Parsed JSON dict, or None if recovery fails
    """
    if not config or not config.get("enabled"):
        return None
    
    try:
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_output_tokens", 500)
        
        # Build the jsonify request
        jsonify_prompt = (
            "You are a JSON formatter. The following is an evaluation response that failed JSON parsing. "
            "Please reformat it into valid JSON with this exact structure:\n"
            "{\n"
            '  "evaluations": [\n'
            "    {\n"
            '      "criterion": "<string>",\n'
            '      "score": <1-5>,\n'
            '      "reason": "<string>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Response to reformat:\n"
            f"{raw_text}\n\n"
            "Return ONLY valid JSON, no markdown or other text."
        )
        
        # Use FilePromptForge to call the jsonify provider
        run_spec = {
            "endpoint": f"{provider}:{model}",
            "input": jsonify_prompt,
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
        runs = [run_spec]
        options = {"json": True}
        
        result = await fpf_runner.run_filepromptforge_batch(runs, options=options)
        
        if result and isinstance(result, list) and len(result) > 0:
            response_text = result[0].get("response", "") if isinstance(result[0], dict) else str(result[0])
            try:
                return json.loads(response_text)
            except Exception:
                # Try to extract JSON from response
                import re as _re
                m = _re.search(r"\{.*\}", response_text, flags=_re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
    except Exception:
        pass
    
    return None


def _aggregate_fpf_costs(log_dir: str, run_group_id: str) -> Dict[str, Any]:
    """
    Sum total_cost_usd across FPF consolidated logs for a given run_group_id.
    Expects logs to be written under log_dir/<run_group_id>/**/*.json.
    """
    summary: Dict[str, Any] = {
        "run_group_id": run_group_id,
        "count": 0,
        "total_cost_usd": 0.0,
        "by_model_provider": {},
        "items": [],
    }
    if not log_dir or not run_group_id:
        return summary

    group_root = os.path.join(log_dir, run_group_id)
    if not os.path.isdir(group_root):
        return summary

    for root, _dirs, files in os.walk(group_root):
        for f in files:
            if not f.lower().endswith(".json"):
                continue
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                # Prefer top-level total_cost_usd; fallback to cost.total_cost_usd
                cost_val = obj.get("total_cost_usd")
                if cost_val is None and isinstance(obj.get("cost"), dict):
                    cost_val = obj["cost"].get("total_cost_usd")
                if cost_val is None:
                    continue
                try:
                    c = float(cost_val)
                except Exception:
                    continue
                summary["total_cost_usd"] += c
                summary["count"] += 1
                # Provider and model are recorded in the consolidated object
                prov = None
                try:
                    cfg = obj.get("config") or {}
                    prov = (cfg.get("provider") or "").strip()
                except Exception:
                    prov = None
                model = obj.get("model")
                key = f"{prov}:{model}"
                summary["by_model_provider"].setdefault(key, 0.0)
                summary["by_model_provider"][key] += c
                summary["items"].append({"path": fp, "total_cost_usd": c})
            except Exception:
                continue

    # Round floats for readability
    summary["total_cost_usd"] = round(float(summary["total_cost_usd"]), 6)
    try:
        summary["by_model_provider"] = {
            k: round(float(v), 6) for k, v in summary["by_model_provider"].items()
        }
    except Exception:
        pass
    return summary


def _default_paths_from_module() -> Tuple[str, str]:
    """
    Resolve default config.yaml and criteria.yaml relative to package root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))  # llm-doc-eval/
    cfg = os.path.join(root, "config.yaml")
    crit = os.path.join(root, "criteria.yaml")
    return cfg, crit


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg_path = config_path or _default_paths_from_module()[0]
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"llm-doc-eval config.yaml not found at: {cfg_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load llm-doc-eval config.yaml but is not installed.")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _collect_models(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    models = (cfg.get("models") or {})
    if not isinstance(models, dict) or not models:
        raise ValueError("config.yaml must define a non-empty 'models' mapping.")
    pairs: List[Tuple[str, str]] = []
    for _, m in models.items():
        if not isinstance(m, dict):
            continue
        provider = (m.get("provider") or "").strip()
        model = (m.get("model") or "").strip()
        if provider and model:
            pairs.append((provider, model))
    if not pairs:
        raise ValueError("config.yaml 'models' has no valid entries (provider+model required).")
    return pairs


def _build_cfg(cfg: Dict[str, Any], provider: str, model: str) -> JudgeCfg:
    jd = cfg.get("judge_defaults") or {}
    llm = cfg.get("llm_api") or {}
    rt = cfg.get("retries") or {}
    return JudgeCfg(
        provider=provider,
        model=model,
        temperature=float(jd.get("temperature", 0.0) or 0.0),
        max_tokens=int(jd.get("max_tokens", 1024) or 1024),
        enable_grounding=bool(jd.get("enable_grounding", False)),
        timeout_seconds=int(llm.get("timeout_seconds", 120) or 120),
        attempts=int(rt.get("attempts", 3) or 3),
        base_delay_seconds=int(rt.get("base_delay_seconds", 2) or 2),
        jitter=bool(rt.get("jitter", True)),
    )


# --- Public API (NO FALLBACKS) -----------------------------------------------

async def run_single_evaluation(
    folder_path: str,
    db_path: Optional[str] = None,
    config_path: Optional[str] = None,
    criteria_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Scan folder_path for candidate reports and run single-document grading via FPF in batch.
    Persists rows into single_doc_results.
    """
    if not folder_path or not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Invalid folder_path: {folder_path}")

    # Determine DB
    db = db_path or DB_PATH
    _ensure_db(db)

    # Discover candidates
    doc_paths = _read_candidates(folder_path)
    if len(doc_paths) < 1:
        return

    # Publish to global DOC_PATHS
    DOC_PATHS.clear()
    DOC_PATHS.update(doc_paths)

    # Load contents
    contents = _load_contents(doc_paths)

    # Load config + criteria
    cfg = _load_config(config_path)
    crit_path = criteria_path or _default_paths_from_module()[1]
    criteria = load_criteria(crit_path)

    # Ensure FPF availability (adds to sys.path if found)
    ensure_fpf(config_path)

    # Models to evaluate
    provider_models = _collect_models(cfg)

    # Resolve concurrency (optional)
    max_conc = None
    try:
        mc = ((cfg.get("llm_api") or {}).get("max_concurrent_llm_calls"))
        if mc is not None:
            max_conc = int(mc)
    except Exception:
        max_conc = None

    # Prepare batch runs
    tmp_dir = tempfile.mkdtemp(prefix="llm_doc_eval_single_batch_")
    # Grouping for FPF logs and cost aggregation (eval → FPF only)
    run_group_id = _uuid.uuid4().hex
    fpf_logs_dir = os.path.join(tempfile.gettempdir(), f"llm_doc_eval_single_logs_{run_group_id}")
    os.makedirs(fpf_logs_dir, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    mapping: Dict[str, Tuple[str, str, str]] = {}  # out_path -> (provider, model, doc_id)

    # Load template once
    here = os.path.dirname(os.path.abspath(__file__))
    single_tpl_path = os.path.abspath(os.path.join(here, "..", "prompts", "single_template.md"))
    if not os.path.exists(single_tpl_path):
        raise FileNotFoundError(f"Single-document prompt template not found: {single_tpl_path}")
    with open(single_tpl_path, "r", encoding="utf-8") as fh:
        single_tpl = fh.read()

    def _sanitize(name: str) -> str:
        import re as _re
        return _re.sub(r"[\\/*?:\"<>| ]+", "_", name or "unknown")

    # Build all runs across models and docs
    for provider, model in provider_models:
        for doc_id in sorted(doc_paths.keys()):
            content = contents.get(doc_id, "")
            crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "- overall quality"
            instr_text = single_tpl.replace("{{CRITERIA}}", crit_lines).replace("{{DOC_CONTENT}}", content)

            # Write files
            uid = (_uuid.uuid4().hex[:8] if _uuid else "uid")
            instr_path = os.path.join(tmp_dir, f"single_{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(doc_id)}_{uid}.txt")
            payload_path = os.path.join(tmp_dir, f"payload_{uid}.txt")
            out_path = os.path.join(tmp_dir, f"out_single_{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(doc_id)}_{uid}.txt")
            try:
                with open(instr_path, "w", encoding="utf-8") as fh:
                    fh.write(instr_text)
                with open(payload_path, "w", encoding="utf-8") as fh:
                    fh.write("Single-document evaluation payload placeholder.")
            except Exception as e:
                raise RuntimeError(f"Failed to write temp instruction/payload files: {e}")

            run_id = f"single-{_sanitize(provider)}-{_sanitize(model)}-{_sanitize(doc_id)}-{uid}"
            runs.append({
                "id": run_id,
                "provider": provider,
                "model": model,
                "file_a": instr_path,
                "file_b": payload_path,
                "out": out_path,
                "overrides": {
                    "request_json": True
                }
            })
            mapping[out_path] = (provider, model, doc_id)

    # Submit in one batch (centralized concurrency inside FPF)
    base_opts: Dict[str, Any] = {"json": True, "run_group_id": run_group_id, "fpf_log_dir": fpf_logs_dir}
    if max_conc is not None:
        base_opts["max_concurrency"] = max_conc
    options = base_opts
    try:
        _ = await fpf_runner.run_filepromptforge_batch(runs, options=options)
    except Exception as e:
        # Proceed to parse any outputs that may have been produced
        # but surface the error if nothing succeeded.
        pass

    summary: Optional[Dict[str, Any]] = None
    conn = sqlite3.connect(db)
    try:
        trial = 1
        # Get jsonify config for recovery fallback
        jsonify_cfg = cfg.get("jsonify") if cfg else None
        
        # Parse outputs
        for out_path, (provider, model, doc_id) in mapping.items():
            if not os.path.exists(out_path):
                continue
            with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            # Try strict JSON first; else find first {...}
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                m = None
                try:
                    import re as _re
                    m = _re.search(r"\{.*\}", raw, flags=_re.DOTALL)
                except Exception:
                    m = None
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None
                
                # If all JSON attempts failed, try jsonify recovery
                if parsed is None and jsonify_cfg and jsonify_cfg.get("enabled"):
                    try:
                        parsed = await _jsonify_response(raw, jsonify_cfg)
                    except Exception:
                        parsed = None
            
            if not isinstance(parsed, dict):
                # No result for this doc
                continue
            evals = parsed.get("evaluations")
            if not isinstance(evals, list) or not evals:
                continue
            model_label = f"{provider}:{model}"
            for item in evals:
                if not isinstance(item, dict):
                    continue
                criterion = item.get("criterion")
                score = item.get("score")
                reason = item.get("reason")
                if not isinstance(criterion, str) or not criterion.strip():
                    continue
                if not isinstance(score, int) or score < 1 or score > 5:
                    continue
                if not isinstance(reason, str) or not reason.strip():
                    continue
                _persist_single_result(conn, doc_id, model_label, trial, criterion.strip(), int(score), reason.strip())
        conn.commit()

        # Aggregate per-runs cost from FPF logs and persist summary
        try:
            summary = _aggregate_fpf_costs(fpf_logs_dir, run_group_id)
            summaries_dir = os.path.join(os.path.dirname(DB_PATH), "fpf_run_summaries")
            os.makedirs(summaries_dir, exist_ok=True)
            out_summary = os.path.join(summaries_dir, f"single_{run_group_id}.json")
            with open(out_summary, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        # Best-effort cleanup
        try:
            for root, _dirs, files in os.walk(tmp_dir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
        except Exception:
            pass
        return summary


async def run_pairwise_evaluation(
    folder_path: str,
    db_path: Optional[str] = None,
    config_path: Optional[str] = None,
    criteria_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Scan folder_path for candidate reports and evaluate all pairs via a single FPF batch per model.
    """
    if not folder_path or not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Invalid folder_path: {folder_path}")

    # Determine DB
    db = db_path or DB_PATH
    _ensure_db(db)

    # Discover candidates
    doc_paths = _read_candidates(folder_path)
    if len(doc_paths) < 2:
        # Nothing to compare; do not write any rows
        return

    # Publish to global DOC_PATHS (contract relied upon by ACM test harness)
    DOC_PATHS.clear()
    DOC_PATHS.update(doc_paths)

    # Load contents
    contents = _load_contents(doc_paths)

    # Load config + criteria
    cfg = _load_config(config_path)
    crit_path = criteria_path or _default_paths_from_module()[1]
    criteria = load_criteria(crit_path)

    # Ensure FPF availability (adds to sys.path if found)
    ensure_fpf(config_path)

    # Evaluate pairs per configured judge models; each model records its own outcomes
    provider_models = _collect_models(cfg)

    # Concurrency option
    max_conc = None
    try:
        mc = ((cfg.get("llm_api") or {}).get("max_concurrent_llm_calls"))
        if mc is not None:
            max_conc = int(mc)
    except Exception:
        max_conc = None

    # Template
    here = os.path.dirname(os.path.abspath(__file__))
    pair_tpl_path = os.path.abspath(os.path.join(here, "..", "prompts", "pairwise_template.md"))
    if not os.path.exists(pair_tpl_path):
        raise FileNotFoundError(f"Pairwise prompt template not found: {pair_tpl_path}")
    with open(pair_tpl_path, "r", encoding="utf-8") as fh:
        pair_tpl = fh.read()

    def _sanitize(name: str) -> str:
        import re as _re
        return _re.sub(r"[\\/*?:\"<>| ]+", "_", name or "unknown")

    # Build all pair combinations once
    pairs = list(itertools.combinations(sorted(doc_paths.keys()), 2))

    summary: Optional[Dict[str, Any]] = None
    conn = sqlite3.connect(db)
    try:
        # Grouping for FPF logs and cost aggregation (eval → FPF only)
        run_group_id = _uuid.uuid4().hex
        fpf_logs_dir = os.path.join(tempfile.gettempdir(), f"llm_doc_eval_pair_logs_{run_group_id}")
        os.makedirs(fpf_logs_dir, exist_ok=True)

        for provider, model in provider_models:
            tmp_dir = tempfile.mkdtemp(prefix="llm_doc_eval_pair_batch_")
            runs: List[Dict[str, Any]] = []
            mapping: Dict[str, Tuple[str, str, str, str]] = {}  # out_path -> (provider, model, a_id, b_id)

            for (a_id, b_id) in pairs:
                doc_a = contents.get(a_id, "")
                doc_b = contents.get(b_id, "")
                crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "- overall quality"
                instr_text = (
                    pair_tpl
                    .replace("{{CRITERIA}}", crit_lines)
                    .replace("{{DOC_A_ID}}", "A")
                    .replace("{{DOC_B_ID}}", "B")
                    .replace("{{DOC_A_CONTENT}}", doc_a)
                    .replace("{{DOC_B_CONTENT}}", doc_b)
                )
                uid = (_uuid.uuid4().hex[:8] if _uuid else "uid")
                instr_path = os.path.join(tmp_dir, f"pair_{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(a_id)}_{_sanitize(b_id)}_{uid}.txt")
                payload_path = os.path.join(tmp_dir, f"payload_{uid}.txt")
                out_path = os.path.join(tmp_dir, f"out_pair_{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(a_id)}_{_sanitize(b_id)}_{uid}.txt")
                try:
                    with open(instr_path, "w", encoding="utf-8") as fh:
                        fh.write(instr_text)
                    with open(payload_path, "w", encoding="utf-8") as fh:
                        fh.write("Pairwise evaluation payload placeholder.")
                except Exception as e:
                    raise RuntimeError(f"Failed to write temp instruction/payload files: {e}")

                run_id = f"pair-{_sanitize(provider)}-{_sanitize(model)}-{_sanitize(a_id)}-{_sanitize(b_id)}-{uid}"
                runs.append({
                    "id": run_id,
                    "provider": provider,
                    "model": model,
                    "file_a": instr_path,
                    "file_b": payload_path,
                    "out": out_path,
                    "overrides": {
                        "request_json": True
                    }
                })
                mapping[out_path] = (provider, model, a_id, b_id)

            # Submit one batch for this provider:model
            base_opts: Dict[str, Any] = {"json": True, "run_group_id": run_group_id, "fpf_log_dir": fpf_logs_dir}
            if max_conc is not None:
                base_opts["max_concurrency"] = max_conc
            options = base_opts
            try:
                _ = await fpf_runner.run_filepromptforge_batch(runs, options=options)
            except Exception:
                # Continue attempting to parse any outputs
                pass

            # Parse results and persist
            model_label = f"{provider}:{model}"
            trial = 1
            for out_path, (_prov, _mod, a_id, b_id) in mapping.items():
                if not os.path.exists(out_path):
                    continue
                with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
                parsed = None
                try:
                    parsed = json.loads(raw)
                except Exception:
                    m = None
                    try:
                        import re as _re
                        m = _re.search(r"\{.*\}", raw, flags=_re.DOTALL)
                    except Exception:
                        m = None
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except Exception:
                            parsed = None
                if not isinstance(parsed, dict):
                    continue
                winner_raw = parsed.get("winner_doc_id")
                reason = parsed.get("reason")
                if not isinstance(winner_raw, str) or winner_raw not in ("A", "B"):
                    continue
                if not isinstance(reason, str) or not reason.strip():
                    continue
                winner_doc_id = a_id if winner_raw == "A" else b_id
                _persist_pair_result(conn, a_id, b_id, winner_doc_id, model_label, trial, reason.strip())

            # Cleanup this tmp_dir
            try:
                for root, _dirs, files in os.walk(tmp_dir, topdown=False):
                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
                try:
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
            except Exception:
                pass
        conn.commit()

        # Aggregate per-runs cost from FPF logs and persist summary
        try:
            summary = _aggregate_fpf_costs(fpf_logs_dir, run_group_id)
            summaries_dir = os.path.join(os.path.dirname(DB_PATH), "fpf_run_summaries")
            os.makedirs(summaries_dir, exist_ok=True)
            out_summary = os.path.join(summaries_dir, f"pair_{run_group_id}.json")
            with open(out_summary, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return summary


async def run_evaluation(
    folder_path: str,
    mode: Optional[str] = "pairwise",
    db_path: Optional[str] = None,
    config_path: Optional[str] = None,
    criteria_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper to run evaluation in single, pairwise, or both modes.
    - mode="single": runs single-document grading only
    - mode="pairwise": runs pairwise comparisons only
    - mode="both": runs single first, then pairwise
    - mode="config" or empty/None: read evaluation.mode from config.yaml (defaults to pairwise)
    """
    # Resolve mode, optionally from config
    m_in = (mode or "").lower().strip()
    if not m_in or m_in == "config":
        cfg = _load_config(config_path)
        m = ((cfg.get("evaluation") or {}).get("mode") or "pairwise").lower().strip()
    else:
        m = m_in

    totals: List[float] = []
    result: Dict[str, Any] = {"mode": m, "total_cost_usd": 0.0}

    if m == "single":
        s = await run_single_evaluation(folder_path=folder_path, db_path=db_path, config_path=config_path, criteria_path=criteria_path)
        if isinstance(s, dict):
            try:
                totals.append(float(s.get("total_cost_usd") or 0.0))
            except Exception:
                pass
    elif m == "pairwise":
        p = await run_pairwise_evaluation(folder_path=folder_path, db_path=db_path, config_path=config_path, criteria_path=criteria_path)
        if isinstance(p, dict):
            try:
                totals.append(float(p.get("total_cost_usd") or 0.0))
            except Exception:
                pass
    elif m == "both":
        s = await run_single_evaluation(folder_path=folder_path, db_path=db_path, config_path=config_path, criteria_path=criteria_path)
        if isinstance(s, dict):
            try:
                totals.append(float(s.get("total_cost_usd") or 0.0))
            except Exception:
                pass
        p = await run_pairwise_evaluation(folder_path=folder_path, db_path=db_path, config_path=config_path, criteria_path=criteria_path)
        if isinstance(p, dict):
            try:
                totals.append(float(p.get("total_cost_usd") or 0.0))
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'single', 'pairwise', 'both', or 'config'.")

    try:
        total_cost = round(sum(totals), 6)
    except Exception:
        total_cost = 0.0
    result["total_cost_usd"] = total_cost
    # Emit a clear final line for downstream capture
    print(f"[EVAL COST] total_cost_usd={total_cost}")
    return result


def get_best_report_by_elo(
    db_path: str,
    doc_paths: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Compute Elo across persisted pairwise_results and return the absolute path
    of the top-ranked document. Falls back to global DOC_PATHS if doc_paths not provided.
    """
    if not db_path or not os.path.exists(db_path):
        # No DB yet or path missing
        return None

    ratings = _compute_elo_from_db(db_path)
    if not ratings:
        return None

    # Pick max Elo
    best_doc_id = max(ratings.items(), key=lambda kv: kv[1])[0]

    paths = doc_paths or DOC_PATHS
    # Resolve path from provided mapping or fallback global
    best_path = paths.get(best_doc_id)
    if best_path and os.path.exists(best_path):
        return best_path

    # As a last resort, search in paths values for matching filename
    # (handles cases where DB doc_id is base filename and mapping keys differ)
    for did, p in paths.items():
        if os.path.basename(did) == os.path.basename(best_doc_id) and os.path.exists(p):
            return p

    return None
