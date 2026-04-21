#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.repo_layout import (
    ANALYSIS_RESULTS_ROOT,
    CANONICAL_DATA_FILE,
    EVALUATION_RESULTS_ROOT,
    MANIFESTS_ROOT,
    MODEL_CONFIG_ROOT,
    PUBLIC_SUMMARY_ROOT,
)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def check_path(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
    }


def check_base_url(base_url: str) -> dict[str, object]:
    candidates = []
    if base_url.endswith("/v1"):
        candidates.append(f"{base_url}/models")
    else:
        trimmed = base_url.rstrip("/")
        candidates.append(f"{trimmed}/models")
        candidates.append(f"{trimmed}/v1/models")

    last_error = None
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(payload)
            return {
                "ok": True,
                "url": url,
                "model_count": len(parsed.get("data", [])) if isinstance(parsed, dict) else None,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last_error = str(exc)

    return {
        "ok": False,
        "url": candidates[-1],
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the K-MetBench private repo environment.")
    parser.add_argument("--base-url", type=str, default="", help="Optional OpenAI-compatible base URL to probe.")
    args = parser.parse_args()

    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "repo": {
            "root": str(REPO_ROOT),
        },
        "modules": {
            "openai": has_module("openai"),
            "dotenv": has_module("dotenv"),
            "tqdm": has_module("tqdm"),
            "torch": has_module("torch"),
            "transformers": has_module("transformers"),
            "PIL": has_module("PIL"),
            "google.genai": has_module("google.genai"),
            "vllm": has_module("vllm"),
        },
        "paths": {
            "explicit_data": check_path(CANONICAL_DATA_FILE),
            "implicit_data": check_path(CANONICAL_DATA_FILE),
            "image_root": check_path(REPO_ROOT / "data" / "shuffled"),
            "canonical_eval_entrypoint": check_path(REPO_ROOT / "scripts" / "eval.py"),
            "model_config_root": check_path(MODEL_CONFIG_ROOT),
            "evaluation_results_root": check_path(EVALUATION_RESULTS_ROOT),
            "analysis_results_root": check_path(ANALYSIS_RESULTS_ROOT),
            "public_summary_root": check_path(PUBLIC_SUMMARY_ROOT),
            "manifest_root": check_path(MANIFESTS_ROOT),
        },
        "env": {
            "GEMINI_API_KEY_set": bool(os.getenv("GEMINI_API_KEY")),
        },
    }
    if args.base_url:
        report["base_url"] = check_base_url(args.base_url)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
