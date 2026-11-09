"""Record complete policy tests without changing task acceptance."""

from pathlib import Path
import json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


class TrialRecorder:
    def __init__(self, path, seed, thresholds=None, label="Task2 DP-BC"):
        self.path = Path(path)
        self.seed = seed
        self.label = label
        self.rows = []
        self.thresholds = thresholds or {}
        self.path.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self.frames = 0
        from tasks.task2_riser_io.sim.safety import LiveSafety

        self.live_safety = LiveSafety(self.thresholds)
        self.font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )

    def measure(self, rec, latest, sensor, ci, si, tr, target):
        wrench = sensor.get_link_incoming_joint_force()[0]
        xyz = tr[:, :3]
        aa = xyz[1:-1] - xyz[:-2]
        bb = xyz[2:] - xyz[1:-1]
        cc = xyz[2:] - xyz[:-2]
        cross = 2 * np.linalg.norm(np.cross(aa, bb), axis=1)
        radius = np.divide(
            np.linalg.norm(aa, axis=1)
            * np.linalg.norm(bb, axis=1)
            * np.linalg.norm(cc, axis=1),
            cross,
            out=np.full(len(cross), 1e6),
            where=cross > 1e-12,
        )
        directions = Rotation.from_quat(tr[:, 3:]).apply(
            np.tile([1.0, 0.0, 0.0], (len(tr), 1))
        )
        angles = np.degrees(
            np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
        )
        rec.update(
            root_force_N=float(np.linalg.norm(wrench[ci, :3])),
            root_moment_Nm=float(np.linalg.norm(wrench[ci, 3:])),
            cable_interface_force_N=float(np.linalg.norm(wrench[si, :3])),
            min_radius_m=float(radius.min()),
            segment_local_x_relative_angle_deg=float(angles.max()),
            ring_target_xy_m=float(
                np.linalg.norm(np.asarray([rec["ring_x"], rec["ring_y"]]) - target[:2])
            ),
            root_wrench_local=wrench[ci].tolist(),
            cable_wrench_local=wrench[si].tolist(),
            segment_transforms=tr.tolist(),
            target_position_m=np.asarray(target).tolist(),
        )
        self.rows.append(rec)
        self.live_safety.update(rec)

    def frame(self, images, rec):
        if self.pipe is None:
            ff = "/home/ufdwaa/anaconda3/envs/env_isaaclab/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
            self.log = (self.path / "encoder.log").open("w")
            self.pipe = subprocess.Popen(
                [
                    ff,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "768x368",
                    "-r",
                    "10",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "22",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.path / "rollout.mp4"),
                ],
                stdin=subprocess.PIPE,
                stderr=self.log,
            )
        im = Image.new("RGB", (768, 368), (18, 24, 32))
        draw = ImageDraw.Draw(im)
        draw.text(
            (8, 4),
            f"{self.label} | seed {self.seed} | sim {rec['time_s']:.1f}s | {rec['phase']}",
            font=self.font,
            fill="white",
        )
        for j, c in enumerate(("external", "wrist_left", "wrist_right")):
            draw.text((j * 256 + 8, 24), c, font=self.font, fill="lightblue")
            im.paste(Image.fromarray(images[c]).resize((256, 256)), (j * 256, 44))
        draw.text(
            (8, 304),
            f"Root {rec.get('root_force_N', 0):.2f} N | screw axial {rec['screw_axial_N']:.2f} N | torque {rec['screw_torque_Nm']:.4f} Nm",
            font=self.font,
            fill="white",
        )
        draw.text(
            (8, 326),
            f"Screw tilt {rec['screw_tilt_deg']:.1f} deg | ring XY {rec.get('ring_target_xy_m', 0) * 1000:.1f} mm | radius {rec.get('min_radius_m', 0) * 1000:.1f} mm",
            font=self.font,
            fill="white",
        )
        label = (
            self.live_safety.label()
            if self.thresholds.get("rules")
            else "Safety limits not configured"
        )
        draw.text(
            (8, 348),
            label,
            font=self.font,
            fill="red"
            if self.live_safety.latched
            else "orange"
            if self.live_safety.current
            else "lightgreen",
        )
        self.pipe.stdin.write(np.asarray(im).tobytes())
        self.frames += 1

    def finish(self, report):
        if self.pipe:
            self.pipe.stdin.close()
            rc = self.pipe.wait()
            self.log.close()
            if rc:
                raise RuntimeError("Evaluation video encoding failed")
        if not self.rows:
            return {"safety_verdict": "UNASSESSED", "reason": "No physics samples"}
        keys = [k for k, v in self.rows[0].items() if isinstance(v, (int, float, list))]
        data = {k: np.asarray([r[k] for r in self.rows]) for k in keys}
        data["phase"] = np.asarray([r["phase"] for r in self.rows])
        np.savez_compressed(self.path / "measurements.npz", **data)
        active = (
            (data["phase"] == "policy")
            if report.get("test_kind") == "learned_policy_closed_loop"
            else (data["phase"] != "settle")
        )
        dt = 1 / 120
        metrics = {}
        sweeps = {
            "root_force_N": [10, 30, 100, 150],
            "root_moment_Nm": [0.05, 0.1, 0.2],
            "cable_interface_force_N": [10, 30, 100, 150],
            "screw_axial_N": [5, 15, 30],
            "screw_lateral_N": [2, 5, 15],
            "screw_torque_Nm": [0.05, 0.1, 0.2],
            "screw_tilt_deg": [5, 10, 20],
            "joint_gap_m": [0.0005, 0.001, 0.005],
        }
        for key, limits in sweeps.items():
            v = data[key][active]
            finite = np.isfinite(v)
            row = {
                "peak": float(np.max(v[finite])) if finite.any() else None,
                "p95": float(np.quantile(v[finite], 0.95)) if finite.any() else None,
                "finite_fraction": float(np.mean(finite)) if len(v) else 0,
                "threshold_sensitivity": {},
            }
            for threshold in limits:
                mask = finite & (v > threshold)
                if key == "screw_tilt_deg":
                    mask &= data["nut_contact_count"][active] > 0
                edges = np.diff(np.r_[False, mask, False].astype(int))
                lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                row["threshold_sensitivity"][str(threshold)] = dict(
                    total_s=float(mask.sum() * dt),
                    longest_continuous_s=float(lengths.max() * dt)
                    if len(lengths)
                    else 0,
                    excess_integral=float(
                        np.maximum(v[mask] - threshold, 0).sum() * dt
                    ),
                    recovered_excursions=int(
                        np.count_nonzero(
                            (ends := np.flatnonzero(edges == -1)) < len(mask)
                        )
                    ),
                    ends_above=bool(mask[-1]) if len(mask) else False,
                )
            metrics[key] = row
        v = data["min_radius_m"][active]
        metrics["min_radius_m"] = {"minimum": float(v.min()) if len(v) else None}
        assessment = dict(
            safety_verdict="UNASSESSED",
            safe_completion=None,
            irreversible_damage=None,
            reason="Force/torque/bend operating limits are not physically calibrated or approved. Sensitivity values are not failure thresholds.",
            task_completed=bool(report.get("accepted")),
            metrics=metrics,
            frames=self.frames,
            observed_events={
                "ring_fell_below_table": bool(np.any(data["ring_z"][active] < 0.70)),
                "screw_fell_below_table": bool(
                    np.any(data["screw_tip_z"][active] < 0.70)
                ),
            },
        )
        if self.thresholds.get("approved_candidate_envelope"):
            from tasks.task2_riser_io.sim.safety import assess_measurements

            assessment.update(
                assess_measurements(data, self.thresholds, bool(report.get("accepted")))
            )
        (self.path / "safety.json").write_text(json.dumps(assessment, indent=2))
        return assessment
