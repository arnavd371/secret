#!/usr/bin/env python3
"""
Runs the offline eval harness and checks it against the stored
regression baseline. CI-shaped: exits 0 on a passing gate (or a
--update-baseline run), exits 1 on a regression. No API key, no
network, no model calls - the harness only exercises the deterministic
decision policy / CAS / grading core (see app/eval/harness.py).

Usage:
    python3 scripts/run_eval.py                  # run + check against baseline
    python3 scripts/run_eval.py --update-baseline # run + overwrite the baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.baseline import DEFAULT_BASELINE_PATH, baseline_exists, load_baseline, save_baseline  # noqa: E402
from app.eval.harness import run_eval_suite  # noqa: E402
from app.eval.regression_gate import check_regression  # noqa: E402


def main() -> int:
    update_baseline = "--update-baseline" in sys.argv

    report = run_eval_suite()
    print(f"Eval suite: {report.total_passed}/{report.total} passed ({report.pass_rate:.1%})")
    for category, rate in report.category_pass_rates().items():
        print(f"  {category:>18}: {rate:.1%}")

    failed = report.failed_cases()
    if failed:
        print("\nFailed cases:")
        for case in failed:
            print(f"  [{case.category.value}] {case.name}: {case.detail}")

    if update_baseline:
        save_baseline(report, DEFAULT_BASELINE_PATH)
        print(f"\nBaseline updated at {DEFAULT_BASELINE_PATH}")
        return 0

    if not baseline_exists(DEFAULT_BASELINE_PATH):
        print("\nNo baseline found - run with --update-baseline first.")
        return 1

    baseline = load_baseline(DEFAULT_BASELINE_PATH)
    gate = check_regression(report, baseline)
    print(f"\n{gate.summary}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
