import numpy as np
import mujoco_py
import RobotCtrl_py
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces



class aliengoEnv(utils.EzPickle):
    def __init__(self):
        model_path = "/home/chenwang/mujoco_aliengo/aliengo_description/xacro/aliengo5.xml"
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None

        utils.EzPickle.__init__(self)
        self.des_vel = [0, 0, 0]  # desired locomotion velocity, vx, vy, yaw_d
        self.rc = RobotCtrl_py.RobotControl()

        # let aliengo stand up
        while not self.rc.is_stand:
            data = self.sim.data
            ctrl = self.rc.run(data.qpos, data.qvel, data.qacc)
            data.ctrl[:12] = ctrl
            self.sim.step()

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()


        print("Initialize Aliengo Environment successfully!")

    def _set_action_space(self):
        low = -np.inf
        high = np.inf
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               self.des_vel])

    #def do_simulation(self, ctrl, n_frames):

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    env = aliengoEnv()
