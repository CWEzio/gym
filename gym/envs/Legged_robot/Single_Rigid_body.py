import gym.envs.Legged_robot.gait as gait
import numpy as np
import time
from scipy import linalg


# from scipy.spatial.transform import Rotation as Rot

def mat_print(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def skew_mat_from_vec(vec):
    if len(vec) != 3:
        raise Exception("The vector length should be 3 when constructing the cross product matrix")
    skew_mat = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return skew_mat


class SRBEnv:
    def __init__(self, env_gait):
        self.hip_loc_body_frame = np.array([[0.19, 0.19, -0.19, -0.19], [-0.114, 0.114, -0.114, 0.114], [0, 0, 0, 0]])
        self.pFoot = np.array([[0.19, 0.19, -0.19, -0.19], [-0.114, 0.114, -0.114, 0.114], [-0.29, -0.29, -0.29, -0.29]])
        self.dt = 0.01
        self.iteration_between_gait_dt = 3
        self.gait = env_gait
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.position = np.zeros((3, 1))
        self.position[2, 0] = 0.29
        self.omega = np.zeros((3, 1))  # angular velocity
        self.velocity = np.zeros((3, 1))
        self.g = 9.81
        self.mass = 9
        self.state_vector = np.zeros((13, 1))
        self.state_vector[12] = self.g
        self.iterator = 0
        self.R_body = np.identity(3)
        self.u = np.zeros((12, 1))
        self.I_body = np.diag([0.07, 0.26, 0.242])
        self.I_world = self.I_body
        self.first_run = True
        self.contact_duration = self.gait.stance * 0.03

    def construct_state_vector(self):
        self.state_vector[0] = self.roll
        self.state_vector[1] = self.pitch
        self.state_vector[2] = self.yaw
        self.state_vector[3: 6] = self.position
        self.state_vector[7: 10] = self.omega
        self.state_vector[10: 13] = self.velocity

    def update_state_from_vector(self):
        self.roll = self.state_vector[0]
        self.pitch = self.state_vector[1]
        self.yaw = self.state_vector[2]
        self.position = self.state_vector[3: 6]
        self.omega = self.state_vector[7: 10]
        self.velocity = self.state_vector[10: 13]

    def rot_mat_x(self):
        rx = np.array(
            [[1, 0, 0], [0, np.cos(self.roll), -np.sin(self.roll)], [0, np.sin(self.roll), np.cos(self.roll)]])
        return rx

    def rot_mat_y(self):
        ry = np.array(
            [[np.cos(self.pitch), 0, np.sin(self.pitch)], [0, 1, 0], [-np.sin(self.pitch), 0, np.cos(self.pitch)]])
        return ry

    def rot_mat_z(self):
        rz = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0], [np.sin(self.yaw), np.cos(self.yaw), 0], [0, 0, 1]])
        return rz

    def calculate_r_body(self):
        self.R_body = self.rot_mat_z().dot(self.rot_mat_y()).dot(self.rot_mat_x())

    def construct_A_mat(self):
        A = np.zeros((13, 13))
        A[: 3, 6: 9] = self.rot_mat_z()
        A[3: 6, 9: 12] = np.identity(3)
        return A

    def construct_B_mat(self):
        B = np.zeros((13, 12))
        r1_skew = skew_mat_from_vec(self.pFoot[:, 0])
        r2_skew = skew_mat_from_vec(self.pFoot[:, 1])
        r3_skew = skew_mat_from_vec(self.pFoot[:, 2])
        r4_skew = skew_mat_from_vec(self.pFoot[:, 3])
        Inertial_inverse = np.linalg.inv(self.I_world)
        temp_mat = np.zeros((3, 12))
        temp_mat[: 3, : 3] = r1_skew
        temp_mat[: 3, 3: 6] = r2_skew
        temp_mat[: 3, 6: 9] = r3_skew
        temp_mat[: 3, 9: 12] = r4_skew
        temp_mat = Inertial_inverse.dot(temp_mat)
        B[6: 9, :] = temp_mat
        eye_over_mass = np.identity(3) / self.mass
        for i in range(4):
            B[9: 12, 3 * i: 3 * (i + 1)] = eye_over_mass
        return B

    def zero_order_discrete(self, A, B):
        assert A.shape == (13, 13)
        assert B.shape == (13, 12)
        ABc = np.zeros((25, 25))
        ABc[0:13, 0:13] = A
        ABc[0:13, 13:25] = B
        ABdt = linalg.expm(ABc * self.dt)
        Adt = ABdt[0:13, 0:13]
        Bdt = ABdt[0:13, 13:25]

        return Adt, Bdt

    # action should be 12 x 1
    def step(self, action):
        self.u = action

        if self.first_run:
            self.gait.reset()
            self.construct_state_vector()
            self.first_run = False

        if self.iterator % self.iteration_between_gait_dt == 0:
            self.gait.step()

        self.calculate_r_body()
        pFoot_ref = self.R_body.dot(self.hip_loc_body_frame) + self.position - np.array([[0], [0], [0.29]])

        contact_phase = self.gait.get_contact_state()
        for leg in range(4):
            if contact_phase[leg] == 0:
                self.u[3*leg: 3*(leg+1), 0] = np.array([0, 0, 0])
                self.pFoot[:, leg] = pFoot_ref[:, leg] + 0.5*(self.velocity[:, 0])*self.contact_duration

        A = self.construct_A_mat()
        B = self.construct_B_mat()
        Adt, Bdt = self.zero_order_discrete(A, B)
        self.state_vector = Adt @ self.state_vector + Bdt @ self.u
        self.update_state_from_vector()
        self.iterator += 1




