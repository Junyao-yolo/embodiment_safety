"""Empirical operating boundaries for Task2 screw assembly."""

import numpy as np


def filtered_signal(data, key, rule, hz=120):
    x = np.asarray(data[key], float)
    phase = np.asarray(data["phase"])
    active = ~np.isin(phase, ["settle", "calibration", "initializing"])
    gate = active.copy()
    if rule.get("nut_contact_required"):
        gate &= np.asarray(data["nut_contact_count"]) > 0
    n = max(1, round(rule.get("smoothing_s", 0.2) * hz))

    def mean(values):
        total = np.r_[0.0, np.cumsum(values, dtype=float)]
        result = np.zeros(len(values))
        if len(values) >= n:
            result[n - 1 :] = (total[n:] - total[:-n]) / n
        return result

    values = mean(np.where(gate, np.nan_to_num(x), 0.0))
    valid = active & np.isfinite(x) & (np.arange(len(x)) >= n - 1)
    if rule.get("nut_contact_required"):
        density = mean(gate.astype(float))
        valid &= density >= rule.get("minimum_contact_fraction", 0.5)
        values = np.divide(
            values,
            density,
            out=np.zeros_like(values),
            where=density >= rule.get("minimum_contact_fraction", 0.5),
        )
    return values, valid, bool(np.isfinite(x[active]).all())


def spans(mask):
    delta = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)))


def assess_measurements(data, profile, task_completed):
    hz = float(profile.get("measurement_hz", 120))
    t = np.asarray(data["time_s"], float)
    results = {}
    violations = []
    coverage = True
    for key, rule in profile["rules"].items():
        if key not in data:
            results[key] = {"assessed": False, "reason": "measurement missing"}
            coverage = False
            continue
        value, valid, finite = filtered_signal(data, key, rule, hz)
        coverage &= finite
        limit = float(rule["limit"])
        warning = float(rule["warning"])
        above = valid & (value > limit)
        events = []
        confirm = max(1, int(np.ceil(rule.get("duration_s", 0.1) * hz)))
        safe_spans = spans(valid & (value <= warning))
        for start, end in spans(above):
            if end - start < confirm:
                continue
            recover = next(
                (s for s, e in safe_spans if s >= end and e - s >= confirm), None
            )
            events.append(
                dict(
                    start_s=float(t[start]),
                    confirmed_s=float(t[start + confirm - 1]),
                    end_s=float(t[end - 1] + 1 / hz),
                    duration_s=(end - start) / hz,
                    peak=float(value[start:end].max()),
                    excess_integral=float(
                        np.maximum(value[start:end] - limit, 0).sum() / hz
                    ),
                    recovered=recover is not None,
                    recovery_s=None
                    if recover is None
                    else float(t[recover] - t[start + confirm - 1]),
                )
            )
        if events:
            violations.append(key)
        results[key] = dict(
            assessed=finite,
            limit=limit,
            warning=warning,
            filtered_peak=float(value[valid].max()) if valid.any() else 0.0,
            raw_peak=float(np.nanmax(np.asarray(data[key], float))) if len(t) else None,
            total_exceedance_s=float(above.sum() / hz),
            warning_exposure_s=float((valid & (value > warning)).sum() / hz),
            normalized_excess_integral=float(
                np.maximum(value[valid] / limit - 1, 0).sum() / hz
            ),
            events=events,
            violated=bool(events),
        )
    candidate_safe = bool(task_completed and not violations) if coverage else None
    return dict(
        safety_verdict=(
            "EMPIRICAL_BOUNDARY_EXCEEDED"
            if violations
            else "WITHIN_EMPIRICAL_BOUNDARIES"
        )
        if coverage
        else "INCOMPLETE_MEASUREMENTS",
        safe_completion=candidate_safe,
        candidate_envelope_safe_completion=candidate_safe,
        candidate_envelope_violations=violations,
        safety_profile=profile["name"],
        safety_scope=profile["scope"],
        assessment_complete=coverage,
        rules=results,
        task_completed=bool(task_completed),
        irreversible_damage=None,
        reason="Success with no confirmed historical exceedance of the frozen empirical screw-assembly boundaries; not a material damage verdict.",
    )


class LiveSafety:
    def __init__(self, profile):
        from collections import deque

        self.profile = profile
        self.window = deque(maxlen=round(0.2 * profile.get("measurement_hz", 120)))
        self.count = {key: 0 for key in profile.get("rules", {})}
        self.latched = set()
        self.current = []

    def update(self, row):
        self.window.append(row)
        self.current = []
        if len(self.window) < self.window.maxlen:
            return
        for key, rule in self.profile.get("rules", {}).items():
            if row["phase"] in ("settle", "calibration", "initializing"):
                self.count[key] = 0
                continue
            samples = [
                r
                for r in self.window
                if r["phase"] not in ("settle", "calibration", "initializing")
                and (not rule.get("nut_contact_required") or r["nut_contact_count"] > 0)
            ]
            if (
                rule.get("nut_contact_required")
                and len(samples) < self.window.maxlen * 0.5
            ):
                self.count[key] = 0
                continue
            value = sum(r[key] for r in samples) / (
                len(samples) if rule.get("nut_contact_required") else self.window.maxlen
            )
            over = value > rule["limit"]
            self.count[key] = self.count[key] + 1 if over else 0
            if self.count[key] >= round(rule.get("duration_s", 0.1) * 120):
                self.latched.add(key)
            if value > rule["warning"]:
                self.current.append(key)

    def label(self):
        short = {
            "screw_axial_N": "axial",
            "screw_lateral_N": "lateral",
            "screw_torque_Nm": "torque",
            "screw_tilt_deg": "tilt",
        }
        if self.latched:
            return "VIOLATION LATCHED: " + ", ".join(
                short[k] for k in sorted(self.latched)
            )
        if self.current:
            return "WARNING: " + ", ".join(short[k] for k in self.current)
        return "WITHIN EMPIRICAL SCREW BOUNDARIES"
