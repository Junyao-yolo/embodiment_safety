"""DAgger mixtures with bounded student segments and measured teacher handback."""
import numpy as np
from feedback import RecoveryRouter

class DaggerRouter(RecoveryRouter):
    def __init__(self,expert,config,seed,policy=None):
        super().__init__(expert,config,seed,policy)
        self.selected="policy";self.pending=[];self.phase=None
        self.phase_blocks=0;self.segment_blocks=0;self.stable_ticks=0
        self.reason="student";self.events=[];self.phase_counts={}
        self.last_observation=None
    def switch(self,kind,reason,phase,observation_id,error):
        if self.selected==kind:return
        self.events.append(dict(observation_id=int(observation_id),phase=phase,
            previous=self.selected,selected=kind,reason=reason,tracking_error_m=float(error)))
        self.selected=kind;self.reason=reason;self.pending=[]
        self.segment_blocks=0;self.stable_ticks=0
        self.policy.clear_history()
        if kind=="expert":self.interventions+=1
    def clear_policy_history(self):
        self.pending=[];self.last_observation=None
        self.policy.clear_history()
    def command(self,state,phase,jacobian,images=None,observation=None,observation_id=None,force_expert=False):
        label=self.expert.query(state)
        error=self.expert.tracking_error(state)
        new=observation_id!=self.last_observation and images is not None
        if phase!=self.phase:
            self.phase=phase;self.phase_blocks=0
            if not self.teacher_override:
                self.switch("policy","phase_start",phase,observation_id,error)
        contact=any(word in phase for word in ("seat","entry","thread","grasp","release"))
        distance=self.config["contact_recovery_distance_m"] if contact else self.config["recovery_distance_m"]
        reason=None
        if force_expert or self.teacher_override:reason=getattr(self,"teacher_override_reason","task_recovery")
        elif error>distance:reason="tracking_departure"
        elif self.selected=="policy" and new and self.segment_blocks>=self.config["student_segment_blocks"]:
            reason="student_segment_budget"
        if reason:self.switch("expert",reason,phase,observation_id,error)
        if self.selected=="expert":
            self.stable_ticks=self.stable_ticks+1 if error<self.config["handback_distance_m"] else 0
            if new and not reason and self.segment_blocks>=self.config["expert_min_blocks"] and self.stable_ticks>=6:
                self.switch("policy","tracking_recovered",phase,observation_id,error)
        if new:
            self.last_observation=observation_id
            self.segment_blocks+=1;self.phase_blocks+=1
            if self.selected=="policy":
                self.pending=list(self.policy.act(images,observation))
        desired=label.copy();source="recovery"
        if self.selected=="policy" and self.pending:
            desired=self.pending.pop(0);source="policy"
        executed=self.expert.limits.commit(desired)
        self.counts[source]+=1
        counts=self.phase_counts.setdefault(phase,{"policy":0,"recovery":0})
        counts[source]+=1
        return label,executed,source
