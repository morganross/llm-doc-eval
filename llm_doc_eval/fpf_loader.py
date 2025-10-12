import os
import sys
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _read_config_fpf_path(config_path: Optional[str]) -> Optional[str]:
    """
    Read fpf.path from a YAML config file if available.
    Returns None if not found/unavailable.
    """
    if not config_path:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            if yaml:
                data = yaml.safe_load(fh) or {}
            else:
                # Minimal YAML parsing fallback: look for "fpf:" block with "path:"
                text = fh.read()
                # very naive scan
                marker = "fpf:"
                idx = text.find(marker)
                if idx == -1:
                    return None
                block = text[idx + len(marker):]
                # look for 'path:' after fpf:
                pidx = block.find("path:")
                if pidx == -1:
                    return None
                # take the rest of the line
                line = block[pidx:].splitlines()[0]
                parts = line.split(":", 1)
                if len(parts) == 2:
                    val = parts[1].strip().strip("'\"")
                    return val or None
                return None
        fpf_cfg = (data or {}).get("fpf") if yaml else None  # type: ignore
        if isinstance(fpf_cfg, dict):
            val = fpf_cfg.get("path")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    except Exception:
        return None


def ensure_fpf(config_path: Optional[str] = None) -> Optional[str]:
    """
    Ensure FilePromptForge is importable by adding its path to sys.path.
    Precedence:
      1) config.yaml's fpf.path if provided and valid
      2) default to sibling ../FilePromptForge relative to this module's directory
    Returns the absolute path used if successful; otherwise None (non-fatal).
    """
    # 1) config-provided path
    cfg_path = _read_config_fpf_path(config_path)
    candidates = []

    if cfg_path:
        # Resolve relative to the config file location if relative
        if not os.path.isabs(cfg_path) and config_path:
            cfg_dir = os.path.dirname(os.path.abspath(config_path))
            candidates.append(os.path.abspath(os.path.join(cfg_dir, cfg_path)))
        else:
            candidates.append(os.path.abspath(cfg_path))

    # 2) sibling default ../FilePromptForge
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(os.path.join(here, "..", "FilePromptForge"))
    candidates.append(sibling)

    for cand in candidates:
        try:
            providers_dir = os.path.join(cand, "providers")
            if os.path.isdir(cand) and os.path.isdir(providers_dir):
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                return cand
        except Exception:
            continue

    # Non-fatal: return None, FPF-backed paths can be skipped until wired
    return None
