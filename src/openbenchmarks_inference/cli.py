"""Shared command line entry point for inference benchmark workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from .workflows.registry import WORKFLOWS, get_workflow


def _workflow_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workflow", choices=tuple(WORKFLOWS), default="micro-single-task")


def _modules(name: str) -> tuple[Any, Any, Any]:
    workflow = get_workflow(name)
    prefix = workflow.__name__
    return (
        import_module(f"{prefix}.lifecycle"),
        import_module(f"{prefix}.runner"),
        import_module(f"{prefix}.watcher"),
    )


def _config(workflow_name: str, spec: Path | None) -> Any:
    workflow = get_workflow(workflow_name)
    return workflow.load_config(spec) if spec is not None else workflow.load_config()


def _run_dir(args: argparse.Namespace, lifecycle: Any, config: Any, environment: dict[str, str]) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    run_dir = lifecycle.initialize_run(spec_path=config.spec_path, environment=environment)
    print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, sort_keys=True), flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openbench-inference")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list-workflows", help="list installed benchmark workflows")

    init = commands.add_parser("init", help="validate inputs and create an immutable local run")
    _workflow_argument(init)
    init.add_argument("--spec", type=Path)
    init.add_argument("--run-id")

    smoke = commands.add_parser("smoke", help="run one 30-request smoke attempt")
    _workflow_argument(smoke)
    smoke.add_argument("--spec", type=Path)
    smoke.add_argument("--run-dir", type=Path)
    smoke.add_argument("--attempt", type=int, choices=(1, 2), default=1)

    run = commands.add_parser("run", help="run the complete benchmark")
    _workflow_argument(run)
    run.add_argument("--spec", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--full", action="store_true", required=True)
    run.add_argument("--approve-full", action="store_true", required=True)
    run.add_argument("--approval-id", required=True)

    watch = commands.add_parser("watch", help="watch durable local run state")
    _workflow_argument(watch)
    watch.add_argument("--run-dir", type=Path, required=True)
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--interval", type=float, default=2.0)

    derive = commands.add_parser("derive-full-run", help="rebuild derived artifacts without network calls")
    _workflow_argument(derive)
    derive.add_argument("--spec", type=Path)
    derive.add_argument("--run-dir", type=Path, required=True)
    derive.add_argument("--approval-id", required=True)
    derive.add_argument("--replace-invalid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-workflows":
        print("\n".join(WORKFLOWS))
        return 0

    lifecycle, runner, watcher = _modules(args.workflow)
    if args.command == "watch":
        watcher.watch(args.run_dir, once=args.once, interval=args.interval)
        return 0

    config = _config(args.workflow, args.spec)
    if args.command == "init":
        run_dir = lifecycle.initialize_run(spec_path=config.spec_path, run_id=args.run_id)
        print(json.dumps({"run_id": run_dir.name, "run_dir": str(run_dir)}, sort_keys=True))
        return 0

    if args.command == "derive-full-run":
        result = asyncio.run(
            runner.derive_full_run(
                args.run_dir,
                approval_id=args.approval_id,
                replace_invalid=args.replace_invalid,
                config=config,
            )
        )
        print(json.dumps({"run_id": args.run_dir.name, "success": result["success"]}, sort_keys=True))
        return 0 if result["success"] else 1

    environment = lifecycle.load_provider_environment(config)
    lifecycle.require_provider_credentials(config, environment)
    run_dir = _run_dir(args, lifecycle, config, environment)
    if args.command == "smoke":
        result = asyncio.run(
            runner.run_smoke(run_dir, args.attempt, environment=environment, config=config)
        )
    else:
        result = asyncio.run(
            runner.run_full(
                run_dir,
                approve_full=args.approve_full,
                approval_id=args.approval_id,
                environment=environment,
                config=config,
            )
        )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
