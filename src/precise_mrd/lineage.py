
"""Lineage tracking utilities.""""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import environment_snapshot, git_sha


@dataclass(slots=True)
class LineageRecord:
    run_id: str
    stage: str
    params: dict[str, Any]
    timestamp: str


class LineageWriter:
    """Persist lineage information for reproducibility.""""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "git_sha": git_sha(),
                "platform": platform.platform(),
                "environment": environment_snapshot(),
                "records": [],
            }

    def append(self, record: LineageRecord) -> None:
        self.payload["records"].append(asdict(record))
        self.path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")


def lineage_record(run_id: str, stage: str, params: dict[str, Any]) -> LineageRecord:
    return LineageRecord(
        run_id=run_id,
        stage=stage,
        params=params,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


