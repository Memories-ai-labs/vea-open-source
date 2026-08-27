"""Structured metric emission.

One record per measurement at a dedicated ``METRIC`` level, so a run's
numbers can be grepped or piped to ``jq`` out of the per-project
``logs/backend.jsonl`` bundle:

    metric("vea.planning.iter.new_clips", 7, iteration=2)

``name`` and ``value`` plus any caller tags ride in the record's ``extra``
dict, which the JSONL handler in ``logging_setup.py`` flattens to
top-level keys.
"""
from __future__ import annotations

import logging
from typing import Any

METRIC_LEVEL = 25  # between INFO (20) and WARNING (30)
logging.addLevelName(METRIC_LEVEL, "METRIC")

logger = logging.getLogger("vea.metrics")


def metric(name: str, value: float, **tags: Any) -> None:
    """Emit one metric record. Never raises — metrics must not break a run."""
    try:
        logger.log(
            METRIC_LEVEL,
            f"{name}={value}" + (f" {tags}" if tags else ""),
            extra={"metric_name": name, "metric_value": value, **tags},
        )
    except Exception:  # noqa: BLE001
        pass
