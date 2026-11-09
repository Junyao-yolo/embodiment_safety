"""Summarize measured Task1 stage transitions."""

import json


def terminal_details(labels):
    final = labels[-1] if labels else {}
    failures = [x["failure"] for x in labels if x.get("failure")]
    return dict(
        stage_failure=failures[-1] if failures else None,
        stage_terminated=bool(final.get("terminated", False)),
        stage_truncated=bool(final.get("truncated", False)),
    )


def label_summary(dest):
    files = sorted(
        dest.glob("stage_transition_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    labels = [json.loads(p.read_text()) for p in files]
    return dict(
        full_transitions=len(labels),
        reward=sum((x["reward"] for x in labels)),
        stages=[e["stage"] for x in labels for e in x["events"]],
        **terminal_details(labels),
    )
