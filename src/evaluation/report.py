from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def save_baseline_report(report: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
