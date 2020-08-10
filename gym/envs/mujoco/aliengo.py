import numpy as np
import mujoco_py
import RobotCtrl_py
import time
from RobotCtrl_py import quat_to_rpy
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
from gym.utils import seeding



# ID of FR Feet is 13, ID of FL Feet is 21, ID of RR Feet is 29, ID of RL Feet is 37


class aliengoEnv(utils.EzPickle):
    def __init__(self):
        model_path = "/home/chenwang/mujoco_aliengo/aliengo_description/xacro/aliengo5.xml"
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        #self.viewer = None
        self.viewer = mujoco_py.MjViewer(self.sim)

        utils.EzPickle.__init__(self)
        self.des_vel = [0, 0, 0]  # desired locomotion velocity, vx, vy, yaw_d
        self.roll_des = 0
        self.pitch_des = 0
        self.height_des = 0.35
        self.dt = 0.002*30  # model's dt times f_ff update frequency

        self.rc = RobotCtrl_py.RobotControl()

        # let aliengo stand up
        while not self.rc.is_stand:
            data = self.sim.data
            ctrl = self.rc.run(data.qpos, data.qvel, data.qacc)
            data.ctrl[:12] = ctrl
            self.sim.step()
            self.viewer.render()

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        print("Initialize Aliengo Environment successfully!")
        print("init_qpos: ", self.init_qpos)
        print("init_qvel: ", self.init_qvel)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        self.rc.reset_controller()
        return ob

    def _feet_contact(self):
        feet_contact = []
        for i in range(4):
            if np.abs(self.sim.data.geom_xpos[13+8*i][2]) < 0.05:
                feet_contact.append(True)
            else:
                feet_contact.append(False)
        return feet_contact

    def _set_action_space(self):
        low = -np.inf
        high = np.inf
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        data = self.sim.data
        feet_contact = self._feet_contact()
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               self.des_vel,
                               feet_contact])

    def do_simulation(self):
        while not self.rc.update_f_ff:
            data = self.sim.data
            ctrl = self.rc.run(data.qpos, data.qvel, data.qacc)
            data.ctrl[:12] = ctrl
            self.sim.step()
            self.viewer.render()

    def step(self, action):
        pos_before = self.sim.data.qpos[:3]
        rpy_before = quat_to_rpy(self.sim.data.qpos[3:7])
        self.rc.set_foot_force(action)
        self.do_simulation()
        pos_after = self.sim.data.qpos[:3]
        rpy_after = quat_to_rpy(self.sim.data.qpos[3:7])
        observation = self._get_obs()

        done = (pos_after[2] < 0.1) or (rpy_after[0] > 0.8) or (rpy_after[1] > 0.8)

        height = pos_after[2]
        roll = rpy_after[0]
        pitch = rpy_after[1]

        vx = (pos_after[0] - pos_before[0]) / self.dt
        vy = (pos_after[1] - pos_before[1]) / self.dt
        yaw_d = (rpy_after[2] - rpy_before[2]) / self.dt

        alive_bonus = 1

        if np.abs(vx - self.des_vel[0]) <= 0.1:
            v_x_reward = np.clip(0.01/np.abs(vx - self.des_vel[0] + 0.0001), 0, 1)
        else:
            v_x_reward = 0

        if np.abs(vy - self.des_vel[1]) <= 0.1:
            v_y_reward = np.clip(0.01/np.abs(vy - self.des_vel[1] + 0.0001), 0, 1)
        else:
            v_y_reward = 0

        if np.abs(yaw_d - self.des_vel[2]) <= 0.05:
            yaw_d_reward = np.clip(0.01/np.abs(yaw_d - self.des_vel[2] + 0.0001), 0, 1)
        else:
            yaw_d_reward = 0

        if np.abs(height - self.height_des) <= 0.05:
            height_reward = np.clip(0.01/np.abs(height - self.height_des + 0.0001), 0, 1)
        else:
            height_reward = 0

        if np.abs(roll - self.roll_des) <= 0.1:
            roll_reward = np.clip(0.01/np.abs(roll - self.roll_des + 0.0001), 0, 1)
        else:
            roll_reward = 0

        if np.abs(pitch - self.pitch_des) <= 0.1:
            pitch_reward = np.clip(0.01/np.abs(pitch - self.pitch_des + 0.0001), 0, 1)
        else:
            pitch_reward = 0

        reward = alive_bonus + v_x_reward + v_y_reward + yaw_d_reward + 0.6*height_reward + 0.6*roll_reward + \
                 0.6*pitch_reward

        info = {
            "position": pos_after,
            "orientation": rpy_after,
            "v_x_reward": v_x_reward,
            "v_y_reward": v_y_reward,
            "yaw_d_reward": yaw_d_reward,
            "height_reward": height_reward,
            "roll_reward": roll_reward,
            "pitch_reward": pitch_reward
        }

        return observation, reward, done, info



    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


if __name__ == "__main__":
    env = aliengoEnv()
    start_time = time.time()
    for _ in range(100):
        env.step(np.zeros(12))
    print("--- %s seconds ---" % (time.time() - start_time))
    env.reset()
    for _ in range(100):
        env.step(np.zeros(12))

