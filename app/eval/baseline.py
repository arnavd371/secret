"""
Baseline persistence: the regression gate needs a stored "last known
good" EvalReport to diff against. Plain JSON on disk, matching this
codebase's no-external-database convention - the baseline is a
committed artifact (app/eval/baseline.json), updated deliberately via
`scripts/run_eval.py --update-baseline` when a change is a real,
reviewed improvement, never auto-overwritten by a normal eval run.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.models import EvalReport

DEFAULT_BASELINE_PATH = Path(__file__).parent / "baseline.json"


def save_baseline(report: EvalReport, path: Path = DEFAULT_BASELINE_PATH) -> None:
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> EvalReport:
    if not path.exists():
        raise FileNotFoundError(
            f"no eval baseline at {path} - run `python3 scripts/run_eval.py --update-baseline` once to create it"
        )
    return EvalReport.model_validate(json.loads(path.read_text()))


def baseline_exists(path: Path = DEFAULT_BASELINE_PATH) -> bool:
    return path.exists()
