"""Central differences with batched rotation residuals at unchanged samples."""
import numpy as np
from scipy.spatial.transform import Rotation

def jacobian(problem, x):
    positions=[]
    rotations=[]
    for k in range(len(x)):
        for sign in (1.,-1.):
            q=x.copy()
            q[k]+=sign*1e-5
            pose=problem.kinematics.pose(q.reshape(-1,1),problem.frame)
            positions.append((problem.base_rotation @ pose.translation + problem.base_position)/problem.units)
            rotations.append(problem.base_rotation @ pose.rotation.matrix())
    dp=np.asarray(positions)-problem.target_position
    dr=(Rotation.from_matrix(np.asarray(rotations))*problem.target_inverse).as_rotvec()
    residuals=np.concatenate((dp*1000.,dr*100.),axis=1)
    problem.residual_calls+=len(positions)
    problem.fk_calls+=len(positions)
    problem.jacobian_calls+=1
    return np.column_stack([(residuals[2*k]-residuals[2*k+1])/2e-5 for k in range(len(x))])
