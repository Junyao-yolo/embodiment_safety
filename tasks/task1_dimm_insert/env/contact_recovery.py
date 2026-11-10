"""Measured contact recovery through the existing velocity controller."""

import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .expert import Task1Expert


class ContactRecovery:
    def __init__(self, panel, spec_path, output):
        self.panel = panel
        self.spec = json.loads(Path(spec_path).read_text())
        self.output = Path(output)
        self.teacher = None
        self.error = None
        self.started = None
        self.wall_start = None
        self.last_info = {}
        self.mode = 'student'
        self.realignments = 0
        self.unsafe_intervals = 0
        self.trace = (self.output / 'teacher_trace.jsonl').open('w', buffering=1)

    def step(self, index, raw, label, cfg):
        if self.teacher is None:
            if index < self.spec['handoff_index']:
                return None
            evidence = (label or {}).get('evidence', {})
            if not all(evidence.get(k) for k in ('grasp_in_allowed_band', 'bilateral_grasp_stable')):
                self.error = 'handoff_grasp_not_stable'
                self.finish(raw, label)
                return None
            self.started = raw['sim_time']
            self.wall_start = time.monotonic()
            self.teacher = Task1Expert(self.panel, self.spec['seed'], variation=0.)
            pos, rotation = self.panel.controller.pose()
            dimm = np.asarray(raw['task_truth']['poses']['dimm'][:3])
            self.teacher.relative = rotation.T @ (dimm - pos)
            self.teacher.change('ALIGN', pos)
            self.teacher.alignz = max(float(dimm[2]), float(self.teacher.entry + .006))
            self.mode = 'clear_contact' if dimm[2] < self.teacher.alignz - .0003 else 'teacher'
            reference = self.spec['reference_raw']
            self.handoff = dict(index=index, sim_time=self.started,
                ee_difference_m=float(np.linalg.norm(np.asarray(raw['ee_position_m'])-reference['ee_position_m'])),
                dimm_difference_m=float(np.linalg.norm(dimm-np.asarray(reference['task_truth']['poses']['dimm'][:3]))),
                actual=raw, reference_episode=self.spec['reference_episode'])
            (self.output / 'teacher_handoff.json').write_text(json.dumps(self.handoff, indent=2))
        if raw['sim_time'] - self.started > self.spec.get('teacher_max_seconds', 120):
            self.error = 'teacher_time_budget'
        if label and not label.get('evidence', {}).get('no_safety_violation', False):
            self.unsafe_intervals += 1
        metrics = self.panel.contact_metrics
        for key in ('finger_left_N', 'finger_right_N', 'dimm_environment_N'):
            if float(metrics.get(key, 0) or 0) > 5.:
                self.error = 'teacher_force_limit:' + key
        if self.error:
            self.finish(raw, label)
            return None
        teacher = self.teacher
        d = np.asarray(raw['task_truth']['poses']['dimm'][:3])
        if teacher.state == 'INSERT':
            angle = (Rotation.from_matrix(teacher.sR) * Rotation.from_quat(raw['task_truth']['poses']['dimm'][3:]).inv()).magnitude()
            if np.linalg.norm(d[:2]-teacher.sp[:2]) > .00035 or angle > np.deg2rad(.25):
                self.realignments += 1
                if self.realignments > 3:
                    self.error = 'repeated_alignment_loss'
                    self.finish(raw, label)
                    return None
                teacher.change('ALIGN', raw['ee_position_m'])
                teacher.alignz = max(float(d[2]), float(teacher.entry+.006))
                self.mode = 'clear_contact'
        action = teacher.action(cfg)
        if teacher.error:
            self.error = teacher.error
            self.finish(raw, label)
            return None
        if self.mode == 'clear_contact':
            action[:6] = 0
            action[2] = np.clip((teacher.alignz-d[2])*.8, 0, .0015)/cfg['translation_m_s']
            if d[2] >= teacher.alignz-.0003:
                self.mode = 'teacher'
                teacher.change('ALIGN', raw['ee_position_m'])
        # Release is permitted only after the existing measured seating/closure gate.
        if teacher.state in ('RELEASE', 'FREE_HOLD', 'RETRACT', 'FINAL_HOLD', 'COMPLETE_PENDING_AUDIT'):
            end = (label or {}).get('end', {})
            if not end.get('seated_operational') or not end.get('closure_history'):
                self.error = 'release_without_verified_seating'
                self.finish(raw, label)
                return None
        reported_stage = self.mode if self.mode=='clear_contact' else teacher.state
        if raw['sim_time']-self.started < .5:
            # Remove the student's outstanding target lead using bounded velocity commands.
            pos, rotation = self.panel.controller.pose()
            goal, goal_rotation = self.panel.controller.goal
            velocity = (np.asarray(pos)-goal)/.1
            velocity *= min(1., .01/max(np.linalg.norm(velocity),1e-12))
            angular = (Rotation.from_matrix(rotation)*goal_rotation.inv()).as_rotvec()/.1
            angular *= min(1., .05/max(np.linalg.norm(angular),1e-12))
            action[:3] = velocity/cfg['translation_m_s']
            action[3:6] = angular/cfg['rotation_rad_s']
            reported_stage = 'settle_goal'
        action = np.clip(action, -1., 1.)
        self.last_info = dict(action_source='teacher', teacher_stage=reported_stage,
            teacher_elapsed_sim_s=raw['sim_time']-self.started, teacher_elapsed_wall_s=time.monotonic()-self.wall_start)
        self.trace.write(json.dumps(dict(index=index, **self.last_info, action=action.tolist(), contact_metrics=metrics), default=float)+'\n')
        return action

    def finish(self, raw, label):
        end = (label or {}).get('end', {})
        completed = bool(self.teacher is not None and (label or {}).get('success') and not self.error)
        report = dict(teacher_started=self.teacher is not None, error=self.error,
            task_completed=completed, training_eligible=completed and self.unsafe_intervals==0,
            unsafe_teacher_intervals=self.unsafe_intervals, realignments=self.realignments,
            elapsed_sim_s=None if self.started is None else raw['sim_time']-self.started,
            elapsed_wall_s=None if self.wall_start is None else time.monotonic()-self.wall_start,
            final_stage=None if self.teacher is None else self.teacher.state, end=end,
            interpretation='recovery_demonstrated' if completed else 'teacher_attempt_unsuccessful_not_proof_of_mechanical_impossibility')
        (self.output / 'teacher_result.json').write_text(json.dumps(report, indent=2))
        return report
