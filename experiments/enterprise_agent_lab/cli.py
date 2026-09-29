"""Command line entry point for the local lab."""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigurationError, load_config
from .runner import run_case


def _summary(trace, trace_path) -> dict:
    decision = trace.final_decision
    return {
        "run_id": trace.run_id,
        "case_id": trace.case_id,
        "condition": trace.condition,
        "mode": trace.mode,
        "model_name": trace.model_name,
        "status": decision.status if decision else None,
        "claims": decision.claims if decision else [],
        "evidence_ids": decision.evidence_ids if decision else [],
        "tool_calls": [call.model_dump(mode="json") for call in trace.tool_calls],
        "policy_checks": [check.model_dump(mode="json") for check in trace.policy_checks],
        "draft_action": (
            decision.draft_action.model_dump(mode="json")
            if decision and decision.draft_action
            else None
        ),
        "trace_path": str(trace_path),
        "error": trace.error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Enterprise Agent Lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run the featured case in replay mode.")
    run_parser = subparsers.add_parser("run", help="Run one case.")
    run_parser.add_argument("--case", default="case-01")
    run_parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    run_parser.add_argument(
        "--condition",
        choices=("raw_schema", "prose_rag", "semantic_catalog"),
        default="semantic_catalog",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate deterministic replay through Agent Improvement Lab.",
    )
    evaluate_parser.add_argument(
        "--condition",
        choices=("raw_schema", "prose_rag", "semantic_catalog"),
        default="semantic_catalog",
    )
    evaluate_parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Evaluate one case. Repeat the option to select several cases.",
    )
    evaluate_parser.add_argument("--repeat", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    try:
        if args.command == "demo":
            trace = run_case("case-01", mode="replay", condition="semantic_catalog", config=config)
        elif args.command == "evaluate":
            from .integrations.agent_improvement import run_replay_evaluation

            result = run_replay_evaluation(
                condition=args.condition,
                case_ids=args.case_ids,
                config=config,
                repeat=args.repeat,
            )
            report = result.report
            print(
                json.dumps(
                    {
                        "run_id": report.run_id,
                        "dataset_id": report.dataset_id,
                        "dataset_version": report.dataset_version,
                        "candidate_id": report.candidate_id,
                        "case_count": len(report.case_results),
                        "passed_count": sum(item.passed for item in report.case_results),
                        "score_count": len(report.scores),
                        "runtime_failures": list(report.runtime_failures),
                        "aggregates": [
                            aggregate.model_dump(mode="json")
                            for aggregate in report.aggregates
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        else:
            trace = run_case(
                args.case,
                mode=args.mode,
                condition=args.condition,
                config=config,
            )
        trace_path = config.run_dir / f"{trace.run_id}.json"
        print(json.dumps(_summary(trace, trace_path), indent=2, sort_keys=True))
        return 0
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
