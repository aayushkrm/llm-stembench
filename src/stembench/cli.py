"""Command-line interface.

Examples:
  stembench run --config configs/stage1_pilot.yaml
  stembench run --config configs/stage1_pilot.yaml --dry-run
  stembench models                     # list free models per provider
  stembench report --run results/stage1/<run_id>   # metrics + tables + figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from stembench.providers.registry import OPENROUTER_FREE, ZEN_FREE
from stembench.schemas import RunConfig


def _cmd_run(args: argparse.Namespace) -> int:
    cfg_dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.dry_run:
        cfg_dict["dry_run"] = True
    config = RunConfig(**cfg_dict)
    from stembench.runner import run

    result = run(config)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    print("openrouter free models:")
    for m in sorted(OPENROUTER_FREE):
        print(f"  {m}")
    print("zen free models:")
    for m in sorted(ZEN_FREE):
        print(f"  {m}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from stembench.report import generate_report

    generate_report(Path(args.run), out_dir=Path(args.out) if args.out else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stembench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run an evaluation from a YAML config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--dry-run", action="store_true",
                       help="use the deterministic fake provider (synthetic records only)")
    run_p.set_defaults(func=_cmd_run)

    models_p = sub.add_parser("models", help="list free models per provider")
    models_p.set_defaults(func=_cmd_models)

    rep = sub.add_parser("report", help="generate metrics, tables, figures for a run")
    rep.add_argument("--run", required=True, help="run directory containing manifest.json")
    rep.add_argument("--out", default=None)
    rep.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
