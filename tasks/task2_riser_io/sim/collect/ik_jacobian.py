"""Analytic Jacobian for the assembly controller's pose residual."""
import numpy as np

def pose_residual_jacobian(kinematics, q, base_rotation, meters_per_unit, error):
    raw = np.asarray(kinematics.jacobian(np.asarray(q, dtype=np.float64).reshape(-1, 1), "panda_hand"))
    x, y, z = error
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    theta2 = float(np.dot(error, error))
    if theta2 < 1e-8:
        coefficient = 1. / 12. + theta2 / 720.
    else:
        theta = np.sqrt(theta2)
        coefficient = (1. - .5 * theta / np.tan(.5 * theta)) / theta2
    inverse_left = np.eye(3) - .5 * skew + coefficient * (skew @ skew)
    return np.vstack((10. * base_rotation @ raw[:3] / meters_per_unit,
                      inverse_left @ base_rotation @ raw[3:],
                      .01 * np.eye(len(q))))
