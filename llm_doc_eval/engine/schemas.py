from typing import List, Dict, Any, Union
import os
import yaml


def load_criteria(criteria_path: str) -> List[str]:
    """
    Load criteria from YAML. Accepts either:
      criteria:
        - factuality
        - relevance
    or
      criteria:
        - name: factuality
          description: ...
          weight: 1.0
        - name: relevance
          description: ...
    Returns a normalized list of criterion names.
    """
    if not criteria_path or not os.path.exists(criteria_path):
        return []
    with open(criteria_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    items = data.get("criteria") or []
    names: List[str] = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            n = item.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
    # dedupe while preserving order
    seen = set()
    result: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def single_doc_schema(criteria: List[str]) -> Dict[str, Any]:
    """
    Provider-agnostic JSON schema dict for single-document evaluation.
    Score range is 1..5 inclusive.
    """
    return {
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"},
                    },
                    "required": ["criterion", "score", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["evaluations"],
        "additionalProperties": False,
    }


def pairwise_schema() -> Dict[str, Any]:
    """
    Provider-agnostic JSON schema dict for pairwise evaluation outcome.
    """
    return {
        "type": "object",
        "properties": {
            "winner_doc_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["winner_doc_id", "reason"],
        "additionalProperties": False,
    }


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal conversion from standard JSON Schema-like dict to a Gemini-compatible
    response_schema structure where types are uppercased identifiers.

    Note: This is a best-effort helper and may require refinement when wiring Gemini.
    """
    def map_type(t: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        if isinstance(t, str):
            m = {
                "string": "STRING",
                "integer": "INTEGER",
                "number": "NUMBER",
                "boolean": "BOOLEAN",
                "object": "OBJECT",
                "array": "ARRAY",
            }
            return m.get(t.lower(), t.upper())
        return t

    def convert(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k == "type":
                    out[k] = map_type(v)
                elif k in ("properties", "items"):
                    out[k] = convert(v)
                else:
                    out[k] = convert(v)
            return out
        if isinstance(node, list):
            return [convert(x) for x in node]
        return node

    return convert(schema)
