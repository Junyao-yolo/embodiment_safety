"""Measured progress and completion for Task2 expert motion segments."""
class MotionTimeout(RuntimeError):
 pass

class MotionProgress:
 """Pause the reference when tracking lags and require measured completion."""
 def __init__(self,steps,settle_steps=0,completion_error=.002):
  self.steps=max(1,int(steps));self.settle_steps=max(0,int(settle_steps))
  self.completion_error=float(completion_error)
  self.cursor=0;self.stable=0;self.elapsed=0
  self.budget=3*(self.steps+self.settle_steps)+180
  self.error=None
 @property
 def index(self):
  return min(self.cursor,self.steps+self.settle_steps-1)
 def update(self,error,velocity):
  self.error=float(error);self.elapsed+=1
  if self.cursor<self.steps:
   if error<.008:self.cursor+=1
  elif self.cursor<self.steps+self.settle_steps:self.cursor+=1
  if self.cursor>=self.steps:
   self.stable=self.stable+1 if error<=self.completion_error and velocity<.08 else 0
  done=self.cursor>=self.steps+self.settle_steps and self.stable>=6
  if not done and self.elapsed>=self.budget:
   raise MotionTimeout(f"Motion did not complete: error={self.error:.6f}, reference={self.cursor}/{self.steps}, ticks={self.elapsed}")
  return done
