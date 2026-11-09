"""Residual and analytic Jacobian for the installed Lula Franka model."""
import numpy as np
from scipy.spatial.transform import Rotation

class PoseResidual:
    def __init__(self, solver, frame, position, rotation):
        self.kinematics = solver._kinematics
        self.frame = frame
        self.base_rotation = solver._robot_rot.copy()
        self.base_position = solver._robot_pos.copy()
        self.units = float(solver._meters_per_unit)
        self.target_position = np.asarray(position)
        self.target_inverse = rotation.inv()
        self.last_x = None
        self.residual_calls = 0
        self.jacobian_calls = 0
        self.fk_calls = 0

    def errors(self, x):
        x = np.asarray(x, dtype=float)
        if self.last_x is None or not np.array_equal(x, self.last_x):
            pose = self.kinematics.pose(x.reshape(-1, 1), self.frame)
            p = (self.base_rotation @ pose.translation + self.base_position) / self.units
            R = self.base_rotation @ pose.rotation.matrix()
            self.dp = p - self.target_position
            self.dr = (Rotation.from_matrix(R) * self.target_inverse).as_rotvec()
            self.last_x = x.copy()
            self.fk_calls += 1
        return self.dp, self.dr

    def __call__(self, x):
        self.residual_calls += 1
        dp, dr = self.errors(x)
        return np.concatenate((dp * 1000., dr * 100.))

    def jacobian(self, x):
        self.jacobian_calls += 1
        _, r = self.errors(x)
        J = self.kinematics.jacobian(np.asarray(x, dtype=float).reshape(-1, 1), self.frame)
        x0, x1, x2 = r
        K = np.array([[0., -x2, x1], [x2, 0., -x0], [-x1, x0, 0.]])
        theta2 = float(r @ r)
        if theta2 < 1e-8:
            coeff = 1./12. + theta2/720. + theta2*theta2/30240.
        else:
            theta = np.sqrt(theta2)
            coeff = (1. - .5*theta/np.tan(.5*theta))/theta2
        log_jacobian = np.eye(3) - .5*K + coeff*(K @ K)
        return np.vstack((1000./self.units*(self.base_rotation @ J[:3]),
                          100.*(log_jacobian @ self.base_rotation @ J[3:])))
