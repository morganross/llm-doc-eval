from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# We rely on ACM's local FilePromptForge runner to make a single model call
# that follows our instructions. This uses a subprocess to call FPF's fpf_main.py.
# Import via fully-qualified package path so it resolves from repo root.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
# Ensure repo root (which contains the 'api_cost_multiplier' package) is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from api_cost_multiplier.functions import fpf_runner  # type: ignore


@dataclass
class JudgeCfg:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 21024
    enable_grounding: bool = True
    timeout_seconds: int = 120
    attempts: int = 3
    base_delay_seconds: int = 2
    jitter: bool = True


class FPFJudge:
    """
    FPF-backed judge backend.

    This implementation invokes FilePromptForge (via ACM's fpf_runner) using a single-shot
    instruction file that:
      - embeds both documents and the criteria list
      - requests STRICT JSON output:
          {"winner_doc_id": "<doc_id>", "reason": "<short explanation>"}

    There is no heuristic fallback. If the model fails to return a valid JSON object,
    this method raises ValueError with a diagnostic. Retries can be added later as needed.
    """

    def __init__(self, fpf_path: Optional[str] = None) -> None:
        self._fpf_path = fpf_path

    def _build_instruction_text(
        self,
        doc1: Dict[str, str],
        doc2: Dict[str, str],
        criteria: List[str],
        cfg: JudgeCfg,
    ) -> str:
        """
        Build the instruction text by loading an external Markdown template and
        substituting placeholders. The LLM must not see original filenames; we
        anonymize as 'A' and 'B' while embedding full contents.
        """
        # Locate the template under ../prompts/pairwise_template.md (relative to this file)
        template_path = os.path.abspath(os.path.join(_HERE, "..", "..", "prompts", "pairwise_template.md"))
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Pairwise prompt template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as fh:
            tpl = fh.read()

        crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "- overall quality"
        doc_a_content = doc1.get("content", "")
        doc_b_content = doc2.get("content", "")

        # Replace placeholders. Always present IDs as 'A' and 'B' to the LLM.
        text = (
            tpl.replace("{{CRITERIA}}", crit_lines)
               .replace("{{DOC_A_ID}}", "A")
               .replace("{{DOC_B_ID}}", "B")
               .replace("{{DOC_A_CONTENT}}", doc_a_content)
               .replace("{{DOC_B_CONTENT}}", doc_b_content)
        )
        return text

    def _parse_first_json_object(self, text: str) -> Dict[str, Any]:
        # Try to find the first {...} JSON object region and parse it.
        # Allow for model to include surrounding text despite instructions.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")
        blob = match.group(0)
        try:
            return json.loads(blob)
        except Exception as e:
            raise ValueError(f"Output looked like JSON but failed to parse: {e}")

    async def generate_single(self, document: str, criteria: List[str], cfg: JudgeCfg) -> Dict[str, Any]:
        """
        Single-document (graded) evaluation using prompts/single_template.md.
        Returns:
          {
            "evaluations": [
              {"criterion": "<name>", "score": 1..5, "reason": "<short explanation>"}
            ]
          }
        """
        if not isinstance(document, str):
            raise ValueError("document must be a string with the full text content.")
        if not isinstance(criteria, list):
            raise ValueError("criteria must be a list of strings.")

        # Load external single-eval template
        template_path = os.path.abspath(os.path.join(_HERE, "..", "..", "prompts", "single_template.md"))
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Single-document prompt template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as fh:
            tpl = fh.read()

        crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "- overall quality"
        instruction_text = (
            tpl.replace("{{CRITERIA}}", crit_lines)
               .replace("{{DOC_CONTENT}}", document)
        )

        # Prepare temp files for the FPF-run (instructions = file_a, payload = file_b)
        tmp_dir = tempfile.mkdtemp(prefix="llm_doc_eval_single_")
        instr_path = os.path.join(tmp_dir, "single_judge_instructions.txt")
        payload_path = os.path.join(tmp_dir, "payload.txt")
        payload_text = "Single-document evaluation payload placeholder."

        try:
            with open(instr_path, "w", encoding="utf-8") as fh:
                fh.write(instruction_text)
            with open(payload_path, "w", encoding="utf-8") as fh:
                fh.write(payload_text)

            options: Dict[str, Any] = {"json": True}

            # Execute a single FPF run
            results: List[Tuple[str, Optional[str]]] = await fpf_runner.run_filepromptforge_runs(
                instr_path, payload_path, num_runs=1, options=options
            )
            if not results:
                raise RuntimeError("FPF did not produce any output for the single judging request.")

            out_path, model_name = results[0]
            if not out_path or not os.path.exists(out_path):
                raise FileNotFoundError("FPF reported success but no output file was found (single).")

            with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()

            data = self._parse_first_json_object(raw)

            # Validate structure
            evals = data.get("evaluations")
            if not isinstance(evals, list) or not evals:
                raise ValueError("Model did not return a non-empty 'evaluations' array.")

            validated: List[Dict[str, Any]] = []
            for idx, item in enumerate(evals):
                if not isinstance(item, dict):
                    raise ValueError(f"evaluations[{idx}] is not an object.")
                criterion = item.get("criterion")
                score = item.get("score")
                reason = item.get("reason")
                if not isinstance(criterion, str) or not criterion.strip():
                    raise ValueError(f"evaluations[{idx}].criterion must be a non-empty string.")
                if not isinstance(score, int) or score < 1 or score > 5:
                    raise ValueError(f"evaluations[{idx}].score must be an integer in [1,5].")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"evaluations[{idx}].reason must be a non-empty string.")
                validated.append({"criterion": criterion.strip(), "score": score, "reason": reason.strip()})

            return {"evaluations": validated}

        finally:
            # Best-effort cleanup of temporary files
            try:
                for p in (instr_path, payload_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                try:
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
            except Exception:
                pass

    async def generate_pairwise(
        self,
        doc1: Dict[str, str],
        doc2: Dict[str, str],
        criteria: List[str],
        cfg: JudgeCfg,
    ) -> Dict[str, Any]:
        if not doc1 or not doc2 or "id" not in doc1 or "id" not in doc2:
            raise ValueError("doc1/doc2 must include 'id' and 'content' fields.")
        if not isinstance(criteria, list):
            raise ValueError("criteria must be a list of strings.")

        # Build instruction content
        instruction_text = self._build_instruction_text(doc1, doc2, criteria, cfg)

        # Prepare temp files for the FPF-run (instructions = file_a, payload = file_b)
        tmp_dir = tempfile.mkdtemp(prefix="llm_doc_eval_judge_")
        instr_path = os.path.join(tmp_dir, "judge_instructions.txt")
        payload_path = os.path.join(tmp_dir, "payload.txt")

        # The payload is not strictly necessary since instructions contain both docs,
        # but FPF's CLI contract requires a file_b; provide a tiny placeholder.
        payload_text = "Pairwise evaluation payload placeholder."

        try:
            with open(instr_path, "w", encoding="utf-8") as fh:
                fh.write(instruction_text)
            with open(payload_path, "w", encoding="utf-8") as fh:
                fh.write(payload_text)

            # Options: allow provider/model override through FPF config if set,
            # or set via options for runner. For now we pass none; FPF will use its config.
            options: Dict[str, Any] = {"json": True}

            # Execute a single FPF run
            results: List[Tuple[str, Optional[str]]] = await fpf_runner.run_filepromptforge_runs(
                instr_path, payload_path, num_runs=1, options=options
            )
            if not results:
                raise RuntimeError("FPF did not produce any output for the judging request.")

            out_path, model_name = results[0]
            if not out_path or not os.path.exists(out_path):
                raise FileNotFoundError("FPF reported success but no output file was found.")

            with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()

            data = self._parse_first_json_object(raw)

            # Validate keys (model must return anonymized 'A' or 'B')
            winner_raw = data.get("winner_doc_id")
            reason = data.get("reason")
            if not isinstance(winner_raw, str) or winner_raw not in ("A", "B"):
                raise ValueError(
                    f"Model returned invalid winner_doc_id: {winner_raw} (expected 'A' or 'B')"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Model returned empty or invalid 'reason'.")

            # Map anonymized winner back to original document ids for persistence
            mapped_winner = doc1["id"] if winner_raw == "A" else doc2["id"]

            return {"winner_doc_id": mapped_winner, "reason": reason.strip()}

        finally:
            # Best-effort cleanup of temporary files
            try:
                for p in (instr_path, payload_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                try:
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
            except Exception:
                # Do not mask prior errors with cleanup issues
                pass
