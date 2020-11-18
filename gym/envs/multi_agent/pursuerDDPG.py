#!/usr/bin/env python3
# In this ddpg trained version, the action will not be mapped using tanh function

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.linalg import solve
from matplotlib.patches import Circle
import numpy as np
import random
import time
import sys

metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
}


# Evader is still


class PursuerDDPG(gym.Env):

    def __init__(self):
        self.kinematics_integrator = 'euler'
        self.viewer = None

        # Pursuers Space
        obs_high = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action_high = np.array([1, 1])
        action_low = np.array([-1, -1])
        pursuer_action_space = spaces.Box(low=action_low, high=action_high)
        pursuer_observation_space = spaces.Box(low=obs_low, high=obs_high)
        self.observation_space = [pursuer_observation_space, pursuer_observation_space]
        self.action_space = [pursuer_action_space, pursuer_action_space]

        self.bound = [obs_low[0], obs_high[0]]

        self.tau = 0.5
        self.e_x = 0.0
        self.e_y = 0.0

        self.state = None
        self.seed()
        self.steps_beyond_done = None
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _out_bound(self, x):
        if x < self.bound[0]:
            return self.bound[0], True
        elif x > self.bound[1]:
            return self.bound[1], True
        else:
            return x, False

    def _check_wall_collision(self, pos):
        pos_x, x_collid = self._out_bound(pos[0])
        pos_y, y_collid = self._out_bound(pos[1])
        return np.array([pos_x, pos_y]), x_collid or y_collid

    def _calc_distance(self, pos1, pos2):
        assert pos1.shape == (2,)
        assert pos2.shape == (2,)
        return np.linalg.norm(pos1 - pos2)

    def step(self, action):
        assert action.shape == (2, 2)
        # Get state data
        # Get pursuers' and evader's params
        p1_x, p1_y, p2_x, p2_y, e_x, e_y = self.state[0]

        # Get action
        # print(self.state)
        p1_v = action[0]
        p2_v = action[1]

        # Pursuer running
        # Pursuer No.1
        p1_old = np.array([p1_x, p1_y])
        dp1 = self.tau * p1_v
        p1 = p1_old + dp1
        # Pursuer N0.2
        p2_old = np.array([p2_x, p2_y])
        dp2 = self.tau * p2_v
        p2 = p2_old + dp2
        # Evader
        e = np.array([self.e_x, self.e_y])

        # Get Last Distance
        last_distance = np.array([self._calc_distance(p1_old, e), self._calc_distance(p2_old, e)])

        # detect collision with the wall
        p1, p1_collid = self._check_wall_collision(p1)
        p2, p2_collid = self._check_wall_collision(p2)

        # Get New Distance
        new_distance = np.array([self._calc_distance(p1, e), self._calc_distance(p2, e)])

        self.state[0] = np.concatenate([p1, p2, e])
        self.state[1] = np.concatenate([p2, p1, e])

        success = (self._calc_distance(p1, e) <= 2 or self._calc_distance(p2, e) <= 2)
        done = success

        # print(distance_difference)
        reward = -new_distance + 1e5 * success
        # reward = -new_distance
        info = {}

        return self.state, reward, done, info

    def reset(self):

        # Initial the high and low
        p_low = np.array([40, 40])
        p_high = np.array([60, 60])
        p1_pos = self.np_random.uniform(p_low, p_high)
        p2_pos = self.np_random.uniform(p_low, p_high)

        self.steps = 0

        # Initial the reset pose
        # evader_state[4] = random.uniform(0.0, 100.0)
        region = random.randint(0, 1)
        if region == 0:
            self.e_x = random.uniform(0.0, 5.0) + 95.0 * random.randint(0, 1)
        else:
            self.e_x = random.uniform(5.0 + 1e-7, 95.0 - 1e-7)
        if self.e_x <= 5.0 or self.e_x >= 95.0:
            self.e_y = random.uniform(0.0, 100.0)
        else:
            self.e_y = random.uniform(0.0, 5.0) + 95.0 * random.randint(0, 1)

        e = np.array([self.e_x, self.e_y])

        self.state = np.array([np.concatenate([p1_pos, p2_pos, e]), np.concatenate([p2_pos, p1_pos, e])])

        self.steps_beyond_done = None

        return self.state

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        screen_width = 400
        screen_height = 400
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.line1 = rendering.Line((100, 100), (100, 300))
            self.line2 = rendering.Line((100, 100), (300, 100))
            self.line3 = rendering.Line((300, 100), (300, 300))
            self.line4 = rendering.Line((300, 300), (100, 300))

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)

            self.pursuer1 = rendering.make_circle(2)
            self.p1trans = rendering.Transform()
            self.pursuer1.add_attr(self.p1trans)
            self.pursuer1.set_color(0, 1, 0)

            self.pursuer2 = rendering.make_circle(2)
            self.p2trans = rendering.Transform()
            self.pursuer2.add_attr(self.p2trans)
            self.pursuer2.set_color(0, 1, 0)

            self.evader = rendering.make_circle(2)
            self.etrans = rendering.Transform()
            self.evader.add_attr(self.etrans)
            self.evader.set_color(0, 0, 1)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.pursuer1)
            self.viewer.add_geom(self.pursuer2)
            self.viewer.add_geom(self.evader)

        if self.state is None:
            return None

        self.p1trans.set_translation(self.state[0][0] * 2 + 100, self.state[0][1] * 2 + 100)
        self.p2trans.set_translation(self.state[1][0] * 2 + 100, self.state[1][1] * 2 + 100)
        self.etrans.set_translation(self.e_x * 2 + 100, self.e_y * 2 + 100)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = PursuerDDPG()
    env.reset()
    no_collid_pos = np.array([50, 50])
    no_collid_pos, _ = env._check_wall_collision(no_collid_pos)
    print('Should print [50, 50]')
    print(no_collid_pos)

    collid_pos1 = np.array([101, 90])
    collid_pos1, _ = env._check_wall_collision(collid_pos1)
    print(('Should print [100, 90]'))
    print(collid_pos1)

    collid_pos2 = np.array([50, -1])
    collid_pos2, _ = env._check_wall_collision(collid_pos2)
    print(('Should print [50, 0]'))
    print(collid_pos2)

    pos1 = np.array([0, 0])
    pos2 = np.array([1, 2])
    print('Should print %.5f' % np.sqrt(5))
    print('%.5f' % env._calc_distance(pos1, pos2))

    print('Print current state of pursuer 1')
    print(env.state[0])
    print('Print current state of pursuer 2')
    print(env.state[1])
    env.e_x = 100
    env.e_y = 100
    for _ in range(500):
        o, r, d, _ = env.step(np.array([[1, 1], [1, 1]]))
        if d:
            break
        env.render()


