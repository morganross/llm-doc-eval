Context (plain English narrative)
- The previous evaluator codebase has been archived into OLD inside api_cost_multiplier/llm-doc-eval. We will rebuild from scratch in this folder so we can iterate cleanly while keeping the legacy code available for reference.
- We are standardizing on FilePromptForge (FPF) as the LLM integration layer because it already encodes deep, battle‑tested knowledge of provider APIs (OpenAI, OpenAI Deep Research, Google Gemini), including structured output patterns, grounding/web search, reasoning controls, and request shaping. We do not want to rebuild that logic ourselves; instead we will reuse FPF and keep our evaluator focused on orchestration (trials/pairs), schema‑validated results, and persistence/summaries.

LLM-Doc-Eval: Detailed Implementation Plan (FPF-Backed)

Executive Summary
- Objective: Rebuild a document evaluator on top of FilePromptForge (FPF) with strict structured outputs, preserving SQLite schema, CLI, and summaries/Elo for compatibility.
- Key Deliverables:
  - fpf_loader.py to resolve/include FPF
  - engine/schemas.py to generate provider-compatible JSON schemas from criteria.yaml
  - engine/judge_backend.py (FPF-backed) to call providers with strict outputs + validation/repair
  - engine/evaluator.py (async orchestration + persistence unchanged)
  - cli.py rewired to new evaluator
  - Updated config.yaml and tests
- Non-goals: Re-implement provider-specific logic that FPF already provides; change DB schema or CLI shape.

1) Architecture and Responsibilities
- Top-level layout (target)
  - llm-doc-eval/
    - fpf_loader.py
    - engine/
      - schemas.py
      - judge_backend.py
      - evaluator.py
      - metrics.py (existing/kept)
      - elo_calculator.py (existing/kept)
    - loaders/
      - text_loader.py (existing/kept)
    - cli.py
    - config.yaml
    - criteria.yaml
    - tests/
    - OLD/ (archived legacy code)

- Module responsibilities
  - fpf_loader.py
    - Resolve FPF location from config.fpf.path or default ../FilePromptForge relative to this folder.
    - Validate FPF presence (providers/ directory); prepend to sys.path once.
    - Expose ensure_fpf(config_path: str) -> str that returns absolute FPF path used.
  - engine/schemas.py
    - Transform criteria.yaml into two JSON Schemas:
      - Single-document scoring: array of {criterion, score, reason}
      - Pairwise: {winner_doc_id, reason}
    - Provide provider-shape adapters if needed (e.g., Gemini type naming).
  - engine/judge_backend.py
    - FPFJudge class to generate judgments via FPF:
      - generate_single(document: str, criteria: list[str], judge_cfg: JudgeCfg) -> dict
      - generate_pairwise(doc1: Doc, doc2: Doc, criteria: list[str], judge_cfg: JudgeCfg) -> dict
    - Enforce strict structured outputs + JSON validation + repair/retries.
    - Support grounding/tooling toggles and reasoning controls through FPF adapters.
  - engine/evaluator.py
    - Async orchestration, concurrency semaphore, retries at request level, persistence into existing SQLite tables.
    - Pair generation for pairwise mode, timestamping, routing to judge backend.
  - cli.py
    - Preserve current command names/flags; route to evaluator; summary/export unchanged.
  - loaders/text_loader.py, engine/metrics.py, engine/elo_calculator.py
    - Kept as-is with minor import path adjustments if needed.

2) Data Model and Persistence (kept)
- Tables (DDL reference; existing DB should already match this shape)

  Single-document
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

  Pairwise
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

- Invariants
  - score ∈ [1,5]
  - winner_doc_id ∈ {doc_id_1, doc_id_2}
  - model records the judge model (e.g., openai:gpt-4.1, gemini:1.5-pro)

3) Configuration Model (config.yaml)
- Example

  fpf:
    path: ""   # optional; absolute or relative to config.yaml; default to ../FilePromptForge if empty

  llm_api:
    max_concurrent_llm_calls: 4
    timeout_seconds: 120

  retries:
    attempts: 3
    base_delay_seconds: 2
    max_delay_seconds: 10
    jitter: true

  judge_defaults:
    temperature: 0.0
    max_tokens: 1024
    enable_grounding: false

  models:
    # Judge A and B are example judge configs used across modes
    model_a:
      provider: openai
      model: gpt-4.1
    model_b:
      provider: google
      model: gemini-1.5-pro

  single_doc_eval:
    trial_count: 3
    criteria_file: criteria.yaml

  pairwise_eval:
    trial_count: 3
    criteria_file: criteria.yaml
    enable_grounding: true

- Validation rules
  - fpf.path: must exist or be empty (fallback to ../FilePromptForge)
  - models entries: provider ∈ {openai, openrouter-openai, openai-deep-research, google}
  - trial_count ≥ 1; max_concurrent_llm_calls ≥ 1
  - timeout, retries sane bounds (nonnegative)

4) Criteria File Format (criteria.yaml)
- Example

  criteria:
    - factuality
    - relevance
    - completeness
    - style_clarity

- Optional extended form (with descriptions and weights)

  criteria:
    - name: factuality
      description: Are statements supported by the provided document?
      weight: 1.0
    - name: relevance
      description: Does the answer focus on the question and ignore irrelevancies?
      weight: 1.0

- The engine tolerates simple string list or extended objects. Stored result schema always includes {criterion, score, reason}.

5) JSON Schema Generation Rules (engine/schemas.py)
- Single-document schema (provider-agnostic; converted if provider needs a different typing vocabulary)

  {
    "type": "object",
    "properties": {
      "evaluations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "criterion": {"type": "string"},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string"}
          },
          "required": ["criterion", "score", "reason"],
          "additionalProperties": false
        }
      }
    },
    "required": ["evaluations"],
    "additionalProperties": false
  }

- Pairwise schema

  {
    "type": "object",
    "properties": {
      "winner_doc_id": {"type": "string"},
      "reason": {"type": "string"}
    },
    "required": ["winner_doc_id", "reason"],
    "additionalProperties": false
  }

- Gemini conversion helper
  - Some Gemini SDKs prefer uppercased type enums or schema objects under response_schema. Provide to_gemini_schema(json_schema: dict) to map string → STRING, object → OBJECT, etc.

6) Public Interfaces and Signatures

- fpf_loader.py

  from pathlib import Path
  import sys, yaml

  def ensure_fpf(config_path: str) -> str:
      """
      Resolve FPF path from config.yaml; fallback to ../FilePromptForge.
      Validate providers/ exists. Prepend to sys.path exactly once.
      Returns absolute path used.
      Raises FileNotFoundError if not found/invalid.
      """

- engine/schemas.py

  from typing import List, Dict, Any

  def load_criteria(criteria_path: str) -> List[str]:
      """Return list of criterion names normalized from simple/extended YAML."""

  def single_doc_schema(criteria: List[str]) -> Dict[str, Any]:
      """Return JSON schema dict for single-document evaluations."""

  def pairwise_schema() -> Dict[str, Any]:
      """Return JSON schema dict for pairwise outcome."""

  def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
      """Convert draft JSON schema to Gemini-compatible schema vocabulary."""

- engine/judge_backend.py

  from dataclasses import dataclass
  from typing import Dict, Any, List

  @dataclass
  class JudgeCfg:
      provider: str          # "openai", "google", "openai-deep-research"
      model: str
      temperature: float = 0.0
      max_tokens: int = 1024
      enable_grounding: bool = False
      timeout_seconds: int = 120
      attempts: int = 3
      base_delay_seconds: int = 2
      jitter: bool = True

  class FPFJudge:
      def __init__(self, fpf_path: str):
          # import FPF modules dynamically, cache adapters

      async def generate_single(self, document: str, criteria: List[str], cfg: JudgeCfg) -> Dict[str, Any]:
          """
          Returns: {"evaluations":[{"criterion":..., "score":..., "reason":...}, ...]}
          - Build schema (single_doc_schema)
          - Route to provider via FPF with strict outputs
          - Validate JSON; on failure, repair prompt + retry with backoff
          """

      async def generate_pairwise(self, doc1: Dict[str, str], doc2: Dict[str, str], criteria: List[str], cfg: JudgeCfg) -> Dict[str, Any]:
          """
          Returns: {"winner_doc_id": <id>, "reason": "..."}
          - Build pairwise schema
          - Route to provider via FPF with strict outputs
          - Validate JSON; on failure repair + retry
          """

- engine/evaluator.py (selected signatures)

  class Evaluator:
      def __init__(self, config_path: str, db_path: str):
          """
          - ensure_fpf(config_path)
          - read config + criteria
          - init FPFJudge
          - init SQLite engine
          - create semaphore with llm_api.max_concurrent_llm_calls
          """

      async def evaluate_single_document(self, doc_id: str, content: str) -> None:
          """Run trials across configured judge models; persist single_doc_results rows."""

      async def evaluate_pairwise_documents(self, docs: Dict[str, str]) -> None:
          """Generate all pairs; run trials per judge; persist pairwise_results rows."""

7) Provider Call Details (via FPF)
- OpenAI (and OpenRouter-OpenAI-compatible)
  - Prefer tool/function calling with JSON Schema:
    - Single: function score_document(parameters=single_doc_schema)
    - Pairwise: function select_winner(parameters=pairwise_schema)
  - Use tool_choice="required" or equivalent strict mode in FPF adapter.
  - Enable web_search_preview if cfg.enable_grounding is true and supported by adapter.
  - Reasoning (effort/limit) exposed via adapter when applicable.

- OpenAI Deep Research (o3/o4-mini deep-research)
  - Similar strict calling via Responses API + background tasks as adapter supports.
  - Polling/long-running behavior encapsulated by FPF; we just pass schema + instructions.

- Google Gemini 1.5/2.x
  - generationConfig.response_mime_type="application/json"
  - generationConfig.response_schema=to_gemini_schema(...)
  - Use tools.google_search for grounding when enabled.
  - Non-streaming for eval (avoid interleaved tokens violating schema).

8) Prompting and Repair Flow
- System/content strategy (provider-neutral instruction sent through FPF):
  - Single-document:
    - Provide the document content and a list of criteria; request strictly JSON matching single_doc_schema.
  - Pairwise:
    - Provide doc A and doc B content (with IDs); request strictly JSON matching pairwise_schema; instruct to select winner_doc_id from the provided IDs.
- On validation failure:
  - Append a repair turn: “Return ONLY valid compact JSON per the schema; no markdown; ensure required fields and no additional properties.”
  - Retry with exponential backoff.

9) Concurrency, Timeouts, and Retries
- Concurrency
  - Outer semaphore in Evaluator sets a cap on concurrent LLM calls; default 4.
- Timeouts
  - Pass timeout_seconds from cfg to FPF adapter if supported; otherwise wrap call in asyncio.wait_for.
- Retry/backoff algorithm
  - attempts=N (default 3)
  - base_delay_seconds=2, exponential growth (2, 4, 8) up to max_delay; jitter if enabled.
  - Retry conditions: schema validation failure, transient HTTP 5xx, provider rate limits/throttling, socket timeouts.

10) Logging and Telemetry
- Default level: INFO
- Structure:
  - Per-call start log: {provider, model, mode: single|pairwise, grounded: bool, attempt: int, doc_ids: [..], criteria_count, request_id}
  - On retry: add {"retry_reason": "...", "delay": seconds}
  - On success: {elapsed_ms, tokens_in?, tokens_out?, cost_estimate? if available from adapter}
- Redaction:
  - Never log API keys or raw prompts/responses.
  - Truncate document previews to first 120 chars; hash doc_id for correlation if needed.
- Verbosity control:
  - config: logging.level (info|debug); debug includes schema payload echoes (still redacted for content).

11) CLI UX and Commands (cli.py)
- Commands (preserve names/shape; examples shown)
  - run-single
    - Usage: python -m llm_doc_eval.cli run-single --config config.yaml --docs ./data/single/
    - Behavior: loads criteria, iterates docs in folder, runs trials across judge models, persists single_doc_results.
  - run-pairwise
    - Usage: python -m llm_doc_eval.cli run-pairwise --config config.yaml --docs ./data/pairwise/
    - Behavior: generates all pairs; persists pairwise_results.
  - summary
    - Usage: python -m llm_doc_eval.cli summary --db results.sqlite --out summary.csv
    - Behavior: compute per-model win rates, Elo; output CSV.
  - export
    - Usage: python -m llm_doc_eval.cli export --db results.sqlite --table single_doc_results --out single.csv
- Flags:
  - --max-concurrent override
  - --attempts / --base-delay override for quick experiments
  - --grounding toggle override per run

12) Testing Strategy
- Unit tests (mock FPF)
  - test_schemas_single_basic
  - test_schemas_pairwise_basic
  - test_judge_single_success_and_retry_on_invalid_json
  - test_judge_pairwise_winner_validation
  - test_evaluator_persistence_single
  - test_evaluator_persistence_pairwise
- Integration smoke (opt-in via env flags)
  - OpenAI: minimal tokens, strict JSON success path
  - Gemini: response_schema success path
  - Grounding on/off toggles
- Fixtures/mocks
  - Fake FPF adapter that returns controlled payloads
  - In-memory SQLite DB
  - Temp directories for docs

13) Performance and Cost Budgets
- Defaults optimized for cost:
  - temperature: 0.0
  - max_tokens: 1024 (tune per criteria verbosity)
  - trial_count default 3; consider 1 for smoke runs
- Metrics to track (debug only if available)
  - per-call latency
  - tokens_in/tokens_out
  - error rate and retry count

14) Security and Compliance
- Secrets via environment variables; no secrets in config.yaml checked into VCS.
- Logging redaction as above; do not store provider raw traces in DB.
- PII: do not include sensitive document content in logs; storage remains user-controlled environment.

15) Migration and Rollout Checklist
- Phase 0: Prework
  - [ ] Ensure FilePromptForge exists adjacent (../FilePromptForge) or set fpf.path
  - [ ] Verify API keys for providers in environment
- Phase 1: Loader + Schemas
  - [ ] Add fpf_loader.py
  - [ ] Implement schemas.py (criteria loading + schema builders + gemini conversion)
  - [ ] Unit test schemas
- Phase 2: Judge Backend
  - [ ] Implement FPFJudge with OpenAI + Gemini initial support
  - [ ] Add validation + repair retry loop
  - [ ] Unit tests for success/retry paths
- Phase 3: Evaluator
  - [ ] Wire Evaluator to FPFJudge; keep persistence identical
  - [ ] Add concurrency semaphore + timeouts
  - [ ] Unit tests for persistence (single + pairwise)
- Phase 4: CLI
  - [ ] Rewire cli.py to new evaluator; retain commands/flags
  - [ ] Smoke test CLI with sample docs
- Phase 5: Verification
  - [ ] Compare summary/Elo vs legacy on a small dataset
  - [ ] Optional integration smokes against real providers (guarded by env flags)
- Phase 6: Docs/Hand-off
  - [ ] Update README with FPF setup notes and config reference
  - [ ] Document logging/telemetry and troubleshooting

16) Risks and Mitigations
- Provider strictness variance (Medium)
  - Mitigation: strict function calling for OpenAI; careful schema; Gemini response_schema conversion helper.
- Grounding tool variability (Medium)
  - Mitigation: config fallback to non-grounded; log grounded flag in metadata.
- Cost/time overruns (Low/Medium)
  - Mitigation: conservative defaults; integration tests gated and minimal; parallelism tuned.
- JSON repair loops masking poor prompts (Low)
  - Mitigation: cap attempts; log final failure with concise diagnostics.

17) Milestones, Timeline, and Definition of Done
- M1 (Day 1-2): fpf_loader + schemas with unit tests (DoD: schema tests pass; FPF path resolution validated)
- M2 (Day 3-4): judge backend (OpenAI + Gemini) with validation/repair (DoD: unit tests for success and invalid JSON repair pass)
- M3 (Day 5): evaluator orchestration + persistence tests (DoD: in-memory SQLite tests pass)
- M4 (Day 6): CLI rewired; local smoke against sample docs (DoD: CSV and DB rows generated)
- M5 (Day 7): Verification vs legacy, docs (DoD: Elo and summaries align within expected variance; README updated)

18) Acceptance Criteria (per deliverable)
- fpf_loader.py: correct path resolution precedence; sys.path updated once; raises on invalid path; unit tests included.
- schemas.py: single/pairwise schemas match specified shapes; Gemini conversion provided; tests cover simple and extended criteria.
- judge_backend.py: strict JSON adherence; retries on invalid JSON; supports grounding toggle; clear error messages; unit tests pass.
- evaluator.py: concurrency respected; persistence shape unchanged; validates winner; captures timestamps; tests pass.
- cli.py: command surfaces unchanged; help text updated; smoke run demonstrates end-to-end persistence.
- config.yaml: validated on load; helpful errors for bad providers/models.
- logging: structured, redacted; debug mode increases detail safely.

19) ACM (api_cost_multiplier) Integration Contract

- Objective: Provide a stable Python API that ACM can import and call to run pairwise evaluation over a folder of candidate reports and then select the best by Elo. Preserve the legacy surface used by api_cost_multiplier/evaluate.py.

- Public module: llm_doc_eval/api.py (back-compat layer)
  - Constants:
    - DOC_PATHS: Dict[str, str] = {}  # doc_id → absolute path of the file evaluated
    - DB_PATH: str = os.path.join(os.path.dirname(__file__), "results.sqlite")
  - Functions:
    - async run_pairwise_evaluation(folder_path: str, db_path: str | None = None, config_path: str | None = None, criteria_path: str | None = None) -> None
      - Enumerate candidate files under folder_path (e.g., *.md, *.txt).
      - Normalize doc_id as the filename (or stable hash if collisions).
      - Populate DOC_PATHS[doc_id] = absolute path discovered in folder_path.
      - Load config (config_path or default ./config.yaml) and criteria (criteria_path or default ./criteria.yaml).
      - Initialize Evaluator (ensure_fpf, schemas, judge backend) and run pairwise trials across configured judge models.
      - Persist rows into db_path (or default DB_PATH).
      - Return None (results persisted; DOC_PATHS populated).
    - get_best_report_by_elo(db_path: str, doc_paths: dict[str, str] | None = None) -> str | None
      - Compute per-document Elo using metrics/elo_calculator from persisted pairwise_results.
      - Determine top-ranked doc_id; map to path via doc_paths or fallback to DOC_PATHS.
      - Return absolute file path of the best candidate (or None if insufficient data).

- Expected ACM usage pattern (current)
  - api_cost_multiplier/evaluate.py:
    - Copies a small set of candidate reports to a temporary directory.
    - Calls run_pairwise_evaluation(folder_path=temp_eval_dir, db_path=DB_PATH).
    - Calls best_path = get_best_report_by_elo(db_path=DB_PATH, doc_paths=DOC_PATHS).
    - Copies best_path into a final output directory for that test.
  - Key assumption preserved:
    - DOC_PATHS is populated by run_pairwise_evaluation as it scans folder_path (the temporary directory in the test harness).
    - get_best_report_by_elo uses either the passed-in doc_paths or the global DOC_PATHS to resolve the winning doc’s path.

- Target ACM usage pattern (pipeline)
  - After runner.py/generate.py produce multiple artifacts for a given input document, ACM may:
    1) Aggregate the produced artifact paths for that input file into an evaluation folder (temporary or permanent).
    2) Invoke run_pairwise_evaluation on that folder with a DB location (shared or per-run).
    3) Invoke get_best_report_by_elo to choose the winner and copy/move it to the final location consumed by downstream steps.
  - CLI alternative (optional):
    - python -m llm_doc_eval.cli run-pairwise --config config.yaml --docs path/to/folder
    - python -m llm_doc_eval.cli summary --db results.sqlite --out summary.csv

- File types and doc_id policy
  - Eligible inputs: .md, .txt by default (extensible via config).
  - doc_id default: base filename (unique within folder_path). If collisions occur, append a short uid or use a stable hash.
  - Winner validation: The evaluator ensures winner_doc_id ∈ {doc_id_1, doc_id_2} for every pair.

- Configuration bridging
  - config.yaml in llm-doc-eval controls judge models and retries; ACM does not need to know provider internals.
  - fpf.path in config.yaml can be set by ACM (or left empty to use ../FilePromptForge).
  - For reproducibility, ACM may pass an explicit db_path per run to avoid cross-run contamination.

- Backward compatibility guarantees
  - Signatures retained: run_pairwise_evaluation(folder_path, db_path) and get_best_report_by_elo(db_path, doc_paths).
  - DOC_PATHS and DB_PATH globals exposed for the test harness and simple scripts.
  - Internal implementation uses FPF-backed evaluator but persists to the same SQLite schema and works with existing summary/Elo code.

- Sample integration snippet (mirrors evaluate.py)

  from llm_doc_eval.api import run_pairwise_evaluation, get_best_report_by_elo, DOC_PATHS, DB_PATH
  import asyncio, shutil, os, uuid

  async def pick_best(candidate_paths: list[str], final_dir: str) -> str | None:
      tmp_dir = os.path.join("temp_llm_eval", str(uuid.uuid4()))
      os.makedirs(tmp_dir, exist_ok=True)
      try:
          for p in candidate_paths:
              shutil.copy2(p, os.path.join(tmp_dir, os.path.basename(p)))
          await run_pairwise_evaluation(folder_path=tmp_dir, db_path=DB_PATH)
          best = get_best_report_by_elo(db_path=DB_PATH, doc_paths=DOC_PATHS)
          if best:
              os.makedirs(final_dir, exist_ok=True)
              shutil.copy2(best, os.path.join(final_dir, os.path.basename(best)))
          return best
      finally:
          shutil.rmtree(tmp_dir, ignore_errors=True)

- Acceptance tests for the contract
  - Given a folder with ≥2 candidate files, run_pairwise_evaluation produces pairwise_results rows and populates DOC_PATHS.
  - get_best_report_by_elo returns a path that exists among the evaluated files.
  - With repeated runs and a clean DB (or isolated db_path), Elo ranking is deterministic given identical evaluator configuration.

Appendix: Developer Notes and Pseudocode

- Single-document judging flow (pseudocode)
  # gather schema
  schema = single_doc_schema(criteria)
  # build prompt/instructions
  sys = "You are a strict evaluator. Return ONLY valid JSON."
  user = f"Document:\\n{content}\\n\\nCriteria:{criteria}"
  # call FPF adapter via provider
  resp = fpf.openai.strict_function_call(
      model=cfg.model,
      system=sys,
      user=user,
      function_name="score_document",
      parameters_schema=schema,
      temperature=cfg.temperature,
      max_tokens=cfg.max_tokens,
      tools=web_search if cfg.enable_grounding else None,
      timeout=cfg.timeout_seconds
  )
  # validate/repair with retries ...

- Pairwise judging flow (pseudocode)
  schema = pairwise_schema()
  sys = "You are an impartial judge. Return ONLY valid JSON."
  user = f"DocA (id={id1}):\\n{content1}\\n\\nDocB (id={id2}):\\n{content2}\\n\\nCriteria:{criteria}"
  resp = provider_call_with_schema(...)
  # validate winner_doc_id in {id1,id2}

Appendix: TODO checklist (expanded granular)
- [ ] Add fpf_loader.py with sibling default and config fpf.path support
- [ ] Implement engine/schemas.py (criteria -> JSON Schemas)
- [ ] Implement engine/judge_backend.py (OpenAI/DP strict functions, Gemini response_schema)
- [ ] Replace evaluator.py to call judge backend and persist unchanged
- [ ] Rewire cli.py; remove Jinja/LangChain direct usage
- [ ] Update config.yaml schema and document fpf.path
- [ ] Unit tests: schema gen, backend retries, evaluator persistence
- [ ] Integration smoke tests across providers (opt-in)
- [ ] Verify summaries and Elo unchanged
- [ ] Write brief README notes for running with FPF
