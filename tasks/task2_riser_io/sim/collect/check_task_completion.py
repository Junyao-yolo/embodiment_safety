"""Numerical diagnostics must not reject a completed physical assembly."""
import unittest
from acceptance import task_assembly_hold,hold_passed
GEOMETRY=dict(seat_radius_m=.0134,ring_radius_m=.011,seat_z_m=[.814,.8195],
 nut_z_m=[.803,.813],screw_head_diameter_m=.017,screw_under_head_z_m=.018)
class TaskCompletionChecks(unittest.TestCase):
 def rows(self):
  r=dict(ring_x=-.092,ring_y=-.071,ring_z=.818,ring_speed_m_s=0.,ring_tilt_deg=0.,
   screw_tip_z=.803,screw_x_m=-.092,screw_y_m=-.071,screw_tilt_deg=0.,
   head_contact_count=1,nut_contact_count=1,head_contact_signed_sum_N=1.,clamp_N=1.,
   robot_joint_positions=[0.]*7+[.03,.03])
  return [r.copy() for _ in range(360)]
 def check(self,rows):return task_assembly_hold(rows,120,[-.092,-.071,.818],GEOMETRY)
 def test_spikes_are_diagnostics(self):
  rows=self.rows()
  rows[30].update(clamp_N=10000.,ring_speed_m_s=50.,ring_tilt_deg=30.)
  r=self.check(rows)
  self.assertTrue(hold_passed(r));self.assertTrue(r["gripper_released"])
  self.assertFalse(r["numerical_quality"]["ring_retained"])
  self.assertGreater(r["mean_clamp_N"],1.)
 def test_terminal_off_seat_fails(self):
  rows=self.rows();rows[-1]["ring_x"]+=.02
  self.assertFalse(hold_passed(self.check(rows)))
 def test_no_nut_contact_is_not_assembled(self):
  rows=self.rows()
  for row in rows:row["nut_contact_count"]=0
  self.assertFalse(hold_passed(self.check(rows)))
 def test_held_part_is_not_released_success(self):
  rows=self.rows();rows[-1]["robot_joint_positions"]=[0.]*9
  self.assertFalse(self.check(rows)["gripper_released"])
if __name__=="__main__":unittest.main()
