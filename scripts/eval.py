#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import (
    main_openai_compatible,
    main_public_openai_compatible,
    main_public_transformers,
    main_transformers,
)
from src.eval.judge import main as main_judge
from src.eval.model_configs import list_model_config_entries, load_model_config
from src.eval.pipeline import PROMPT_CHOICES
from src.eval.public_protocol import PUBLIC_DATA_CHOICES, PUBLIC_PROMPT_CHOICES

OPENAI_COMPATIBLE_ENGINES = {"api", "vllm"}
TRANSFORMERS_ENGINES = {"transformers"}


def _add_direct_runner_args(parser: argparse.ArgumentParser, *, backend: str) -> None:
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--explicit-data-file", type=str, default=None)
    parser.add_argument("--implicit-data-file", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8192/v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-type", type=str, default="advanced", choices=PROMPT_CHOICES)
    parser.add_argument("--data-type", type=str, default="explicit", choices=["explicit", "implicit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if backend == "openai_compatible":
        parser.add_argument("--concurrency", type=int, default=1)
    elif backend == "transformers":
        parser.add_argument("--device", type=str, default="cuda")
    else:  # pragma: no cover
        raise ValueError(f"Unsupported backend: {backend}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified K-MetBench evaluation entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation.")
    run_parser.add_argument("--model-config", type=str, default=None, help="Model config path or identifier under configs/models/.")
    run_parser.add_argument("--list-model-configs", action="store_true", help="List available model configs and exit.")
    run_parser.add_argument("--protocol", choices=["private", "public"], default="private")
    run_parser.add_argument("--prompt-type", type=str, default="advanced", choices=PROMPT_CHOICES)
    run_parser.add_argument("--data-type", type=str, default="explicit", choices=["explicit", "implicit"])
    run_parser.add_argument("--image-root", type=str, default=None)
    run_parser.add_argument("--explicit-data-file", type=str, default=None)
    run_parser.add_argument("--implicit-data-file", type=str, default=None)
    run_parser.add_argument("--output-root", type=str, default=None)
    run_parser.add_argument("--api-key", type=str, default=None)
    run_parser.add_argument("--base-url", type=str, default=None)
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--top-p", type=float, default=None)
    run_parser.add_argument("--max-tokens", type=int, default=None)
    run_parser.add_argument("--num-samples", type=int, default=-1)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--concurrency", type=int, default=None)
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true", help="Print the resolved dispatch payload without running evaluation.")

    judge_parser = subparsers.add_parser("judge", help="Judge reasoning outputs from a reasoning run.")
    judge_parser.add_argument("--model", type=str, required=True)
    judge_parser.add_argument("--predictions", type=str, default=None)
    judge_parser.add_argument("--evaluator", type=str, default="gemini-2.5-pro")
    judge_parser.add_argument("--base-url", type=str, default="https://generativelanguage.googleapis.com/v1beta/openai/")
    judge_parser.add_argument("--api-key", type=str, default="")
    judge_parser.add_argument("--explicit-data-file", type=str, default=None)
    judge_parser.add_argument("--output-root", type=str, default=None)
    judge_parser.add_argument("--data-type", choices=["explicit"], default="explicit")
    judge_parser.add_argument("--prompt-type", choices=["reasoning_evaluation"], default="reasoning_evaluation")
    judge_parser.add_argument("--num-samples", type=int, default=-1)
    judge_parser.add_argument("--seed", type=int, default=42)
    judge_parser.add_argument("--temperature", type=float, default=0.0)
    judge_parser.add_argument("--top-p", type=float, default=0.95)
    judge_parser.add_argument("--timeout", type=int, default=300)
    judge_parser.add_argument("--max-parse-retries", type=int, default=3)
    judge_parser.add_argument("--wo-rationale", action="store_true")
    judge_parser.add_argument("--quiet", action="store_true")

    run_api_parser = subparsers.add_parser(
        "run-api",
        help="Run private OpenAI-compatible evaluation with a direct model name instead of a model config.",
    )
    _add_direct_runner_args(run_api_parser, backend="openai_compatible")

    run_transformers_parser = subparsers.add_parser(
        "run-transformers",
        help="Run private transformers evaluation with a direct model name instead of a model config.",
    )
    _add_direct_runner_args(run_transformers_parser, backend="transformers")
    return parser


def _render_available_model_configs() -> str:
    return "\n".join(label for label, _ in list_model_config_entries())


def _resolve_api_key(env_name: str | None, override: str | None) -> str:
    if override is not None:
        return override
    if not env_name or env_name.upper() == "EMPTY":
        return ""
    return os.getenv(env_name, "")


def _validate_protocol(args: argparse.Namespace) -> None:
    if args.protocol == "public":
        if args.prompt_type not in PUBLIC_PROMPT_CHOICES:
            allowed = ", ".join(PUBLIC_PROMPT_CHOICES)
            raise SystemExit(f"Public protocol only supports prompt types: {allowed}")
        if args.data_type not in PUBLIC_DATA_CHOICES:
            allowed = ", ".join(PUBLIC_DATA_CHOICES)
            raise SystemExit(f"Public protocol only supports data types: {allowed}")


def _build_dispatch_argv(args: argparse.Namespace, *, model_name: str, engine: str, base_url: str | None, api_key: str) -> list[str]:
    argv = [
        "--model",
        model_name,
        "--prompt-type",
        args.prompt_type,
        "--data-type",
        args.data_type,
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
    ]
    if args.image_root:
        argv.extend(["--image-root", args.image_root])
    if args.explicit_data_file:
        argv.extend(["--explicit-data-file", args.explicit_data_file])
    if args.implicit_data_file and args.protocol == "private":
        argv.extend(["--implicit-data-file", args.implicit_data_file])
    if args.output_root:
        argv.extend(["--output-root", args.output_root])
    if api_key:
        argv.extend(["--api-key", api_key])
    if base_url:
        argv.extend(["--base-url", base_url])
    if args.temperature is not None:
        argv.extend(["--temperature", str(args.temperature)])
    if args.top_p is not None:
        argv.extend(["--top-p", str(args.top_p)])
    if args.max_tokens is not None:
        argv.extend(["--max-tokens", str(args.max_tokens)])
    if args.quiet:
        argv.append("--quiet")
    if engine in OPENAI_COMPATIBLE_ENGINES and args.concurrency is not None:
        argv.extend(["--concurrency", str(args.concurrency)])
    if engine in TRANSFORMERS_ENGINES and args.device is not None:
        argv.extend(["--device", args.device])
    return argv


def _run_command(args: argparse.Namespace) -> int:
    _validate_protocol(args)
    if not args.model_config:
        raise SystemExit("Missing --model-config. Use --list-model-configs to inspect available entries.")

    config = load_model_config(args.model_config)
    engine = config.model.engine.lower()
    api_key = _resolve_api_key(config.model.api_key_env, args.api_key)
    base_url = args.base_url if args.base_url is not None else config.model.base_url
    resolved_params = config.resolve_params(args.prompt_type)

    if args.temperature is None and resolved_params.temperature is not None:
        args.temperature = float(resolved_params.temperature)
    if args.top_p is None and resolved_params.top_p is not None:
        args.top_p = float(resolved_params.top_p)
    if args.max_tokens is None and resolved_params.max_tokens is not None:
        args.max_tokens = int(resolved_params.max_tokens)
    if args.device is None:
        transformers_section = config.get_section("transformers")
        if isinstance(transformers_section.get("device"), str):
            args.device = str(transformers_section["device"])

    dispatch_argv = _build_dispatch_argv(
        args,
        model_name=config.model.name,
        engine=engine,
        base_url=base_url,
        api_key=api_key,
    )

    if engine in OPENAI_COMPATIBLE_ENGINES:
        target = main_public_openai_compatible if args.protocol == "public" else main_openai_compatible
        target_name = "openai_compatible"
    elif engine in TRANSFORMERS_ENGINES:
        target = main_public_transformers if args.protocol == "public" else main_transformers
        target_name = "transformers"
    else:
        raise SystemExit(f"Unsupported engine for unified eval entrypoint: {config.model.engine}")

    if args.dry_run:
        payload = {
            "model_config": config.relative_path,
            "model_name": config.model.name,
            "client": config.model.client,
            "engine": config.model.engine,
            "protocol": args.protocol,
            "dispatch_target": target_name,
            "argv": dispatch_argv,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    return target(dispatch_argv)


def _build_direct_dispatch_argv(args: argparse.Namespace, *, backend: str) -> list[str]:
    argv = [
        "--model",
        args.model,
        "--prompt-type",
        args.prompt_type,
        "--data-type",
        args.data_type,
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
    ]
    if args.image_root:
        argv.extend(["--image-root", args.image_root])
    if args.explicit_data_file:
        argv.extend(["--explicit-data-file", args.explicit_data_file])
    if args.implicit_data_file:
        argv.extend(["--implicit-data-file", args.implicit_data_file])
    if args.output_root:
        argv.extend(["--output-root", args.output_root])
    if args.api_key:
        argv.extend(["--api-key", args.api_key])
    if args.base_url:
        argv.extend(["--base-url", args.base_url])
    if args.temperature is not None:
        argv.extend(["--temperature", str(args.temperature)])
    argv.extend(["--top-p", str(args.top_p)])
    if args.max_tokens is not None:
        argv.extend(["--max-tokens", str(args.max_tokens)])
    if args.quiet:
        argv.append("--quiet")
    if backend == "openai_compatible":
        argv.extend(["--concurrency", str(args.concurrency)])
    elif backend == "transformers":
        argv.extend(["--device", args.device])
    else:  # pragma: no cover
        raise ValueError(f"Unsupported backend: {backend}")
    return argv


def _run_direct_command(args: argparse.Namespace, *, backend: str) -> int:
    dispatch_argv = _build_direct_dispatch_argv(args, backend=backend)
    if backend == "openai_compatible":
        target = main_openai_compatible
        target_name = "openai_compatible_direct"
    elif backend == "transformers":
        target = main_transformers
        target_name = "transformers_direct"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported backend: {backend}")

    if args.dry_run:
        payload = {
            "model_name": args.model,
            "backend": backend,
            "dispatch_target": target_name,
            "argv": dispatch_argv,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    return target(dispatch_argv)


def _normalize_argv(argv: list[str]) -> list[str]:
    if argv and argv[0].startswith("-"):
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.list_model_configs:
            rendered = _render_available_model_configs()
            if rendered:
                print(rendered)
                return 0
            raise SystemExit("No model configs found under configs/models/")
        return _run_command(args)
    if args.command == "run-api":
        return _run_direct_command(args, backend="openai_compatible")
    if args.command == "run-transformers":
        return _run_direct_command(args, backend="transformers")
    if args.command == "judge":
        return main_judge(argv[1:])
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
