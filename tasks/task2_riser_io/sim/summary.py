"""Keep task outcome, empirical safety and execution errors distinct."""

from collections import Counter
import json
from pathlib import Path

INVALID_ACTIONS = ("Command outside joint or gripper range", "Nonfinite action")


def classify_trial(report, safety, trial, seed, wall_s, simulation_s, out):
    error = report.get("error")
    valid = (
        report.get("command_blocks", 0) > 0 or error in INVALID_ACTIONS
    ) and error in (None, "released assembly incomplete", *INVALID_ACTIONS)
    success = bool(report.get("accepted") and report.get("thread_engagement_verified"))
    if not valid:
        category = "execution_error"
    elif success:
        category = None
    elif error in INVALID_ACTIONS:
        category = "invalid_policy_action"
    elif not report.get("wire_lifted"):
        category = "initial_grasp"
    elif not report.get("wire_seated"):
        category = "wire_transfer_or_seating"
    elif not report.get("screw_lifted"):
        category = "screw_pickup"
    elif not report.get("screw_fastened") or not report.get(
        "thread_engagement_verified"
    ):
        category = "screw_fastening"
    else:
        category = "release_or_retention"
    return dict(
        trial=trial,
        seed=seed,
        valid_policy_test=valid,
        task_success=success,
        failure_category=category,
        wall_s=wall_s,
        simulation_s=simulation_s,
        report=report,
        safety=safety,
        video=str(out / "rollout.mp4"),
        measurements=str(out / "measurements.npz"),
    )


def summarize(results, output_dir):
    valid = [row for row in results if row["valid_policy_test"]]
    n = len(valid)
    success = sum(row["task_success"] for row in valid)
    safe = sum(row["safety"].get("safe_completion") is True for row in valid)
    exceeded = sum(
        bool(row["safety"].get("candidate_envelope_violations")) for row in valid
    )
    coverage = sum(row["safety"].get("assessment_complete") is True for row in valid)
    summary = dict(
        completed=len(results),
        valid_trials=n,
        execution_errors=len(results) - n,
        task_successes=success,
        task_success_rate=success / n if n else None,
        safe_completions=safe,
        safe_completion_rate=safe / n if n and coverage == n else None,
        boundary_violation_rate=exceeded / n if n and coverage == n else None,
        safety_assessed_trials=coverage,
        damage_rate=None,
        safety_scope="Frozen empirical screw assembly operating boundaries; material damage uncalibrated",
        failure_categories=dict(
            Counter(row["failure_category"] for row in valid if not row["task_success"])
        ),
        wall_seconds=sum(row["wall_s"] for row in results),
    )
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
